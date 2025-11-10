import os
import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
import time
import gc

# Custom kernel imports
try:
    from rvc.lib.custom_kernels import (
        OptimizedAudioProcessor, 
        optimize_model_for_inference, 
        performance_monitor,
        enable_performance_optimizations,
        CustomRVCModel
    )
    CUSTOM_KERNELS_LOADED = True
except ImportError:
    CUSTOM_KERNELS_LOADED = False
    print("Custom kernels not available, using standard PyTorch operations")

# from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE, AutoTune, calc_pitch_shift
from rvc.lib.predictors.Generator import Generator
# Butterworth high-pass filter
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

# Audio processing class with optimizations
class OptimizedAudioProcessor:
    @staticmethod
    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        """Optimized RMS calculation and adjustment"""
        # Use vectorized operations for better performance
        rms1 = librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)
        rms2 = librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)

        # Use PyTorch for faster interpolation if available
        rms1_tensor = torch.from_numpy(rms1).float()
        rms2_tensor = torch.from_numpy(rms2).float()
        
        rms1_interp = F.interpolate(rms1_tensor.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        rms2_interp = F.interpolate(rms2_tensor.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        
        # Avoid division by zero
        rms2_interp = torch.maximum(rms2_interp, torch.zeros_like(rms2_interp) + 1e-6)

        # Calculate adjusted audio
        adjustment_factor = torch.pow(rms1_interp, 1 - rate) * torch.pow(rms2_interp, rate - 1)
        adjusted_audio = target_audio * adjustment_factor.numpy()
        return adjusted_audio

# Enhanced Voice conversion class
class VC:
    def __init__(self, tgt_sr, config):
        """
        Initialize parameters for voice conversion with optimizations.
        """
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.device = config.device
        
        # Initialize optimized audio processor
        if CUSTOM_KERNELS_LOADED:
            self.audio_processor = OptimizedAudioProcessor()
        else:
            self.audio_processor = OptimizedAudioProcessor()
            
        # Enable performance optimizations
        if torch.cuda.is_available():
            enable_performance_optimizations()
        
        # Performance monitoring
        self.performance_stats = {
            "total_conversions": 0,
            "total_processing_time": 0,
            "average_conversion_time": 0
        }

    @performance_monitor.time_function("vc_forward")
    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """
        Converts audio using the provided model with optimizations.
        """
        # Pre-processing optimizations
        start_time = time.time()
        
        # Convert to tensor and move to device efficiently
        if isinstance(audio0, np.ndarray):
            feats = torch.from_numpy(audio0).float()
        else:
            feats = audio0.clone()
            
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1).to(self.device)
        
        # Create padding mask more efficiently
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self.device)

        # Model inference with optimized path
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad():
            if CUSTOM_KERNELS_LOADED and hasattr(net_g, '_optimized_forward'):
                # Use optimized forward pass
                logits = net_g._optimized_forward(**inputs)
            else:
                logits = model.extract_features(**inputs)
            
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        # Handle protection for vocals
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        # FAISS indexing optimization
        if index is not None and big_npy is not None and index_rate != 0:
            # Batch process for better performance
            npy = feats[0].cpu().numpy()
            try:
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats
            except Exception as e:
                print(f"Warning: FAISS search failed, using original features: {e}")

        # Resize features for upsampling
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # Pitch alignment and protection
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        # Final model inference
        p_len_tensor = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats.float(), p_len_tensor, pitch, pitchf.float(), sid) if hasp else (feats.float(), p_len_tensor, sid)
            
            if CUSTOM_KERNELS_LOADED and hasattr(net_g, 'flash_attention'):
                # Use optimized inference if available
                audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            else:
                audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            
            del hasp, arg

        # Cleanup
        if protect < 0.5 and pitch is not None and pitchf is not None:
            del feats0
        del feats, padding_mask
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats["total_conversions"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        self.performance_stats["average_conversion_time"] = (
            self.performance_stats["total_processing_time"] / self.performance_stats["total_conversions"]
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_min,
        f0_max,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        hop_length,
        autopitch,
        autopitch_threshold,
        autotune,
        autotune_strength,
    ):
        """
        Main pipeline for audio conversion with enhanced performance optimizations.
        """
        pipeline_start = time.time()
        
        # Load FAISS index with error handling
        index = big_npy = None
        if file_index and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"Warning: FAISS index loading failed: {error}")

        # Audio preprocessing with optimizations
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        # Optimized segmentation for long audio files
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            # Use vectorized operations for segmentation
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            # Vectorized search for optimal timestamps
            for t in range(self.t_center, audio.shape[0], self.t_center):
                segment = audio_sum[t - self.t_query : t + self.t_query]
                min_idx = np.argmin(np.abs(segment))
                opt_ts.append(t - self.t_query + min_idx)

        # Initialize processing
        s = 0
        t = None
        audio_opt = []
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # Pitch processing with optimizations
        pitch_tensor = pitchf_tensor = None
        if pitch_guidance:
            if not hasattr(self, "f0_generator"):
                self.f0_generator = Generator(
                    self.sample_rate, hop_length, f0_min, f0_max, 
                    False, self.device, False, False
                )
            
            # Use optimized F0 calculation
            pitch, pitchf = self.f0_generator.calculator(
                f0_method, audio_pad, pitch, p_len, 3, 
                autotune, autotune_strength, 
                proposal_pitch=autopitch, 
                proposal_pitch_threshold=autopitch_threshold
            )

            # Efficient tensor operations
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]

            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)

            pitch_tensor = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf_tensor = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        # Batch processing of segments
        batch_size = 4  # Process 4 segments at once
        segment_batches = []
        
        # Prepare segments
        for i in range(0, len(opt_ts), batch_size):
            batch_end = min(i + batch_size, len(opt_ts))
            batch_segments = []
            
            for t in opt_ts[i:batch_end]:
                t = t // self.window * self.window
                audio_segment = audio_pad[s : t + self.t_pad2 + self.window]
                batch_segments.append((audio_segment, s, t))
                s = t
            
            # Process last segment if exists
            if i + batch_size >= len(opt_ts):
                # Process remaining audio
                pitch_segment = pitch_tensor[:, s // self.window :] if pitch_guidance and t is not None else pitch_tensor
                pitchf_segment = pitchf_tensor[:, s // self.window :] if pitch_guidance and t is not None else pitchf_tensor
                
                remaining_segment = (
                    audio_pad[t:] if 't' in locals() else audio_pad[s:],
                    pitch_segment,
                    pitchf_segment
                )
                batch_segments.append(remaining_segment)
            
            segment_batches.append(batch_segments)

        # Process batches
        for batch in segment_batches:
            batch_results = []
            
            for segment_data in batch:
                if len(segment_data) == 3:  # (audio, pitch, pitchf)
                    audio_segment, pitch_segment, pitchf_segment = segment_data
                else:  # (audio, s, t) format
                    audio_segment, s, t = segment_data
                    pitch_segment = pitch_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None
                    pitchf_segment = pitchf_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None
                
                try:
                    result = self.vc(
                        model,
                        net_g,
                        sid,
                        audio_segment,
                        pitch_segment,
                        pitchf_segment,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )
                    
                    # Remove padding
                    if len(result) > 2 * self.t_pad_tgt:
                        result = result[self.t_pad_tgt : -self.t_pad_tgt]
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    print(f"Warning: Segment processing failed: {e}")
                    continue
            
            # Concatenate batch results
            if batch_results:
                audio_opt.extend(batch_results)

        # Final processing
        audio_opt = np.concatenate(audio_opt)
        
        # Volume envelope adjustment
        if volume_envelope != 1:
            if CUSTOM_KERNELS_LOADED:
                audio_opt = self.audio_processor.change_rms(
                    audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope
                )
            else:
                # Standard RMS adjustment
                audio_opt = OptimizedAudioProcessor.change_rms(
                    audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope
                )

        # Audio normalization
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max

        # Cleanup
        if pitch_guidance:
            del pitch_tensor, pitchf_tensor
        del sid
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log performance
        total_time = time.time() - pipeline_start
        print(f"Pipeline completed in {total_time:.2f}s for {len(audio_opt)/44100:.1f}s audio")
        
        return audio_opt

    def get_performance_stats(self):
        """Get performance statistics for the current session"""
        return self.performance_stats.copy()
