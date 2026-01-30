from typing import Dict, Any, List
import os
from src.config import CONFIG
from src.utils import get_logger
from src.audio.processing import load_audio
from src.asr.mock_asr import MockASR
from src.asr.whisper_asr import WhisperASR
from src.diarization.pyannote_diarizer import PyannoteDiarizer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.intent import IntentClassifier
from src.analysis.tone import ToneAnalyzer
from src.analysis.ner import NERExtractor
from src.scoring.engine import ScoringEngine
from src.reporting.exporter import Exporter

logger = get_logger(__name__)

class AudioPipeline:
    def __init__(self):
        self.asr = None
        self.diarizer = None
        self.sentiment = None
        self.intent = None
        self.tone = None
        self.ner = None
        self._loaded = False

    def load_resources(self):
        if self._loaded: return
        
        logger.info("Initializing pipeline resources...")
        
        # ASR
        if CONFIG.use_api == "whisper_local":
            self.asr = WhisperASR()
            self.diarizer = PyannoteDiarizer()
            self.diarizer.load()
        elif CONFIG.use_api == "mock":
            self.asr = MockASR()
            self.diarizer = None
        else:
            # Fallback to mock for now if others not implemented
            logger.warning(f"ASR method '{CONFIG.use_api}' not fully implemented, using Mock.")
            self.asr = MockASR()
            self.diarizer = None

        # Analysis
        self.sentiment = SentimentAnalyzer(CONFIG.sentiment_model)
        self.sentiment.load()
        
        self.intent = IntentClassifier() # uses default BART
        self.intent.load()
        
        self.tone = ToneAnalyzer(CONFIG.speech_emotion_model)
        self.tone.load()
        
        self.ner = NERExtractor(CONFIG.ner_model)
        self.ner.load()
        
        self._loaded = True

    def process_file(self, audio_path: str, out_dir: str = "outputs") -> Dict[str, Any]:
        if not self._loaded:
            self.load_resources()

        if not os.path.exists(audio_path) and CONFIG.use_api != "mock":
            logger.error(f"File not found: {audio_path}")
            return {}

        logger.info(f"Processing {audio_path}...")
        
        # 1. Transcribe
        transcript, asr_turns = self.asr.transcribe(audio_path)
        
        # 2. Diarize & Align (if supported)
        final_turns = asr_turns
        diarized_segments = []
        
        if self.diarizer and CONFIG.use_api == "whisper_local":
            logger.info("Starting diarization...")
            diarized_segments = self.diarizer.diarize(audio_path)
            logger.info(f"Diarization finished. Found {len(diarized_segments)} segments.")
            if not diarized_segments or (len(diarized_segments) == 1 and diarized_segments[0]["speaker"] == "speaker_unknown"):
                 logger.warning("Diarization returned no segments or unknown. Alignment will likely fail/fallback.")
            
            final_turns = self._align_speakers(asr_turns, diarized_segments)
            
            # Reconstruct transcript with speakers
            transcript = "\n".join([f'[{t["speaker"].upper()}] {t["text"]}' for t in final_turns])
            
        elif CONFIG.use_api == "mock":
            # Mock already returns perfect turns
            # We can synthesize diarized segments from it for completeness
            diarized_segments = [{"speaker": t["speaker"], "start": t["start"], "end": t["end"]} for t in final_turns]
            # Run alignment to populate confidence and roles
            final_turns = self._align_speakers(final_turns, diarized_segments)

        # 3. Analysis
        logger.info("Running analysis...")
        sentiment_res = self.sentiment.analyze(transcript)
        tone_res = self.tone.analyze(audio_path) if CONFIG.use_api != "mock" else {"label": "UNKNOWN", "score": 0.0}
        
        # Intent uses transcript or diarized text
        intent_res = self.intent.classify(transcript)
        
        # NER
        entities_res = self.ner.extract(transcript)
        
        # 4. Scoring
        scores_res = ScoringEngine.score(final_turns, intent_res, sentiment_res, tone_res)
        
        # 5. Export
        Exporter.save(
            transcript=transcript,
            turns=final_turns,
            diarized=diarized_segments,
            intent=intent_res,
            entities=entities_res,
            scores=scores_res,
            sentiment=sentiment_res,
            speech_emotion=tone_res,
            out_dir=out_dir
        )
        
        return {
            "transcript": transcript,
            "turns": final_turns,
            "diarized": diarized_segments,
            "intent": intent_res,
            "entities": entities_res,
            "scores": scores_res,
            "sentiment": sentiment_res,
            "speech_emotion": tone_res
        }

    def _align_speakers(self, asr_segments: List[Dict], diarized_segments: List[Dict]) -> List[Dict]:
        """
        Align ASR text segments with Diarization speaker segments based on overlap.
        Calculates a 'confidence' score for the attribution based on overlap ratio.
        """
        assigned = []
        for t in asr_segments:
            t_start = t["start"]
            t_end = t["end"]
            t_dur = t_end - t_start
            
            if t_dur <= 0:
                continue

            # Find all speakers overlapping with this segment
            # Score = overlapping duration
            speaker_scores = {}
            for d in diarized_segments:
                # intersection
                overlap = max(0.0, min(t_end, d["end"]) - max(t_start, d["start"]))
                if overlap > 0:
                    speaker_scores[d["speaker"]] = speaker_scores.get(d["speaker"], 0.0) + overlap
            
            # Find best speaker
            if not speaker_scores:
                best_label = "speaker_unknown"
                confidence = 0.0
            else:
                best_label = max(speaker_scores, key=speaker_scores.get)
                total_overlap = speaker_scores[best_label]
                # Confidence = coverage of the ASR segment by this speaker
                # (capped at 1.0)
                confidence = min(1.0, total_overlap / t_dur)

            assigned.append({
                "speaker": best_label,
                "start": t_start,
                "end": t_end,
                "text": t["text"],
                "confidence": round(confidence, 2)
            })
            
        # Merge adjacent turns if same speaker
        merged = []
        for seg in assigned:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                # Merge if same speaker AND gap is small (< 1.0s)
                if prev["speaker"] == seg["speaker"] and (seg["start"] - prev["end"] <= 1.0):
                    # Extend end
                    prev["end"] = seg["end"]
                    # Concatenate text
                    prev["text"] = (prev["text"].strip() + " " + seg["text"].strip()).strip()
                    # Average confidence (weighted by duration would be better, but simple avg is fine for now)
                    prev["confidence"] = round((prev["confidence"] + seg["confidence"]) / 2, 2)
                else:
                    merged.append(seg)
        
        # Assign Roles (Collector vs Debtor)
        return self._assign_roles(merged)

    def _assign_roles(self, turns: List[Dict]) -> List[Dict]:
        """
        Assign roles using Zero-Shot Classification (ML-Based).
        Aggregates text per speaker and classifies as 'Collector' or 'Debtor'.
        """
        if not turns:
            return []
            
        # 1. Aggregate text per speaker
        speaker_texts = {}
        for t in turns:
            spk = t.get("speaker")
            if not spk or spk in ["speaker_unknown", "UNKNOWN"]:
                continue
            speaker_texts.setdefault(spk, []).append(t.get("text", ""))
            
        # 2. Prepare for Classification
        # We reuse the IntentClassifier's loaded pipeline if available
        classifier = self.intent.classifier if self.intent else None
        
        if not classifier:
            logger.warning("Zero-shot classifier not available. Roles will be undetermined.")
            # Return with unknown roles
            return self._mark_all_unknown(turns)
            
        # 3. Classify each speaker
        # Labels: Use descriptive natural language labels for the model
        candidate_labels = [
            "Debt collection agent calling about a loan",
            "Customer responding about repayment"
        ]
        
        speaker_scores = {}
        
        for spk, texts in speaker_texts.items():
            full_text = " ".join(texts)
            if not full_text.strip():
                continue
                
            try:
                # Truncate text if too long (BART has 1024 token limit, ~4000 chars safe bet)
                input_text = full_text[:4000]
                
                result = classifier(input_text, candidate_labels=candidate_labels, multi_label=False)
                
                # Extract score for "Debt collection agent..."
                # result['labels'] and result['scores'] are sorted by score
                agent_label = "Debt collection agent calling about a loan"
                
                if result['labels'][0] == agent_label:
                    score = result['scores'][0]
                else:
                    # If agent label is second, score is lower (implicit logic)
                    # We can map it: score for agent = score if label=agent else (1-score) approx
                    # Better: find index
                    idx = result['labels'].index(agent_label)
                    score = result['scores'][idx]
                    
                speaker_scores[spk] = score
                
            except Exception as eobj:
                logger.warning(f"Classification failed for {spk}: {eobj}")
                speaker_scores[spk] = 0.5

        logger.info(f"ML Role Scores (Agent Probability): {speaker_scores}")
        
        # 4. Determine Roles based on scores
        # Highest score > 0.6 is Collector? Or just relative comparison?
        # User requirement: "Assign COLLECTOR to speaker with higher agent probability"
        
        collector_id = None
        if speaker_scores:
            # Sort by agent probability descending
            sorted_spks = sorted(speaker_scores.items(), key=lambda x: x[1], reverse=True)
            candidate, score = sorted_spks[0]
            
            # If valid comparison (more than 1 speaker), taking top as collector is safe
            # If single speaker, need threshold
            if len(speaker_scores) > 1:
                collector_id = candidate
            elif len(speaker_scores) == 1:
                if score > 0.6: # Reasonable confidence for single speaker
                    collector_id = candidate

        # 5. Assign to Turns
        final_turns = []
        for t in turns:
            spk = t.get("speaker")
            
            if not spk or spk not in speaker_scores:
                 t["role"] = None
                 t["speaker_id"] = spk
                 t["speaker"] = f"{spk} (undetermined)" if spk else "SPEAKER_UNKNOWN"
            elif spk == collector_id:
                t["role"] = "collector"
                t["speaker_id"] = spk
                t["speaker"] = "COLLECTOR"
            else:
                 # If we found a collector, others are debtors
                 if collector_id:
                     t["role"] = "debtor"
                     t["speaker_id"] = spk
                     t["speaker"] = "DEBTOR"
                 else:
                     # No confident collector
                     t["role"] = None
                     t["speaker_id"] = spk
                     t["speaker"] = f"{spk} (undetermined)"
            
            final_turns.append(t)
            
        return final_turns

    def _mark_all_unknown(self, turns):
        for t in turns:
            t["role"] = None
            t["speaker"] = f"{t.get('speaker', 'SPEAKER')} (undetermined)"
        return turns
