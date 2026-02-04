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
            
            # Preprocess for Diarization: Ensure 16kHz Mono
            # Pyannote is sensitive to sample rate and format
            import tempfile
            import soundfile as sf
            
            try:
                # Load using our robust loader
                wav, sr = load_audio(audio_path, target_sr=16000)
                
                # Create temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    
                sf.write(tmp_path, wav, sr)
                logger.info(f"Created temp 16kHz mono audio for diarization: {tmp_path}")
                
                # Run Diarization on clean audio
                diarized_segments = self.diarizer.diarize(tmp_path)
                
                # Cleanup
                os.remove(tmp_path)
                
            except Exception as e:
                logger.error(f"Audio preprocessing for diarization failed: {e}. Falling back to original file.")
                diarized_segments = self.diarizer.diarize(audio_path)

            # Raw Logging for Debugging
            unique_speakers = set(s["speaker"] for s in diarized_segments)
            logger.info(f"[DIARIZATION_RAW] Segments: {len(diarized_segments)}")
            logger.info(f"[DIARIZATION_RAW] Speakers Found: {unique_speakers}")
            logger.info(f"[DIARIZATION_RAW] Raw Output Sample: {diarized_segments[:5]}")
            
            # Validate diarization output
            if not diarized_segments:
                error_msg = "DIARIZATION FAILED: No segments returned by Pyannote"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Log speaker count (but don't fail on single speaker)
            if len(unique_speakers) < 2:
                logger.warning(f"[DIARIZATION_WARNING] Only {len(unique_speakers)} speaker(s) detected. Role assignment will be skipped.")
                logger.warning(f"Detected speakers: {unique_speakers}")
            else:
                logger.info(f"[DIARIZATION_SUCCESS] {len(unique_speakers)} speakers detected: {unique_speakers}")
            
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
        
        # DEBUG: Log unique speakers before returning
        unique_speakers_in_turns = set(t.get("speaker", "MISSING") for t in final_turns)
        logger.info(f"[PIPELINE_OUTPUT] Unique speakers in final_turns: {unique_speakers_in_turns}")
        logger.info(f"[PIPELINE_OUTPUT] Total turns: {len(final_turns)}")
        
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
        Constraint: Assign strictly based on max time overlap.
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
                # No overlap - find nearest diarization segment instead of defaulting to unknown
                if diarized_segments:
                    # Find nearest segment by time
                    nearest_seg = min(diarized_segments, 
                                     key=lambda d: min(abs(d["start"] - t_start), abs(d["end"] - t_end)))
                    best_label = nearest_seg["speaker"]
                    confidence = 0.5  # Lower confidence for nearest-match
                    logger.debug(f"No overlap for ASR segment [{t_start:.2f}-{t_end:.2f}], using nearest speaker: {best_label}")
                else:
                    best_label = "speaker_unknown"
                    confidence = 0.0
            else:
                best_label = max(speaker_scores, key=speaker_scores.get)
                total_overlap = speaker_scores[best_label]
                # Confidence = coverage of the ASR segment by this speaker
                confidence = min(1.0, total_overlap / t_dur)

            assigned.append({
                "speaker": best_label,
                # Keep original ASR times for the text
                "start": t_start,
                "end": t_end,
                "text": t["text"],
                "confidence": round(confidence, 2)
            })
            
        # Merge adjacent turns if same speaker (to handle aggregation)
        merged = []
        for seg in assigned:
            if not merged:
                merged.append(seg)
            else:
                prev = merged[-1]
                # Merge if same speaker AND small gap
                if prev["speaker"] == seg["speaker"] and (seg["start"] - prev["end"] <= 2.0):
                    prev["end"] = seg["end"]
                    prev["text"] = (prev["text"].strip() + " " + seg["text"].strip()).strip()
                    prev["confidence"] = round((prev["confidence"] + seg["confidence"]) / 2, 2)
                else:
                    merged.append(seg)
        
        # DEBUG: Log speakers after alignment and merging
        unique_after_merge = set(s.get("speaker", "MISSING") for s in merged)
        logger.info(f"[ALIGNMENT_DEBUG] After merge - Unique speakers: {unique_after_merge}")
        logger.info(f"[ALIGNMENT_DEBUG] After merge - Sample turns: {merged[:3]}")
        
        # Apply role assignment as post-processing
        return self._assign_roles(merged)

    def _assign_roles(self, turns: List[Dict]) -> List[Dict]:
        """
        Pure post-processing role assignment using weighted keyword scoring.
        ALWAYS assigns COLLECTOR and DEBTOR when ≥2 speakers.
        """
        if not turns:
            return turns
        
        logger.info("[ROLE_ASSIGNMENT] Starting post-processing role assignment...")
        
        # Step 1: Aggregate ALL text per speaker (read-only)
        speaker_texts = {}
        speaker_first_appearance = {}  # Track order of first appearance
        
        for turn in turns:
            speaker = turn.get("speaker", "")
            if not speaker or speaker == "speaker_unknown":
                continue
            
            # Track first appearance order
            if speaker not in speaker_first_appearance:
                speaker_first_appearance[speaker] = turn.get("start", 0)
            
            text = turn.get("text", "")
            if speaker not in speaker_texts:
                speaker_texts[speaker] = ""
            speaker_texts[speaker] += " " + text
        
        # Clean up aggregated text
        for speaker in speaker_texts:
            speaker_texts[speaker] = speaker_texts[speaker].strip()
        
        logger.info(f"[ROLE_ASSIGNMENT] Aggregated text for {len(speaker_texts)} speaker(s)")
        
        # If not exactly 2 speakers, skip role assignment
        if len(speaker_texts) != 2:
            logger.info(f"[ROLE_ASSIGNMENT] Expected 2 speakers, found {len(speaker_texts)}. Skipping role assignment.")
            for turn in turns:
                turn["role"] = None
                turn["speaker_id"] = turn.get("speaker", "speaker_unknown")
            return turns
        
        # Step 2: Weighted keyword scoring
        collector_keywords = {
            "calling from": 2,
            "bank": 2,
            "loan": 2,
            "emi": 2,
            "due date": 2,
            "payment reminder": 2,
            "this call is recorded": 2
        }
        
        debtor_keywords = {
            "i will pay": 2,
            "salary": 2,
            "next week": 2,
            "tomorrow": 2,
            "cannot pay": 2,
            "give time": 2
        }
        
        speaker_scores = {}
        
        for speaker, text in speaker_texts.items():
            text_lower = text.lower()
            
            collector_score = 0
            debtor_score = 0
            
            # Score collector keywords
            for keyword, weight in collector_keywords.items():
                if keyword in text_lower:
                    collector_score += weight
                    logger.info(f"[ROLE_ASSIGNMENT] {speaker}: Found collector keyword '{keyword}' (+{weight})")
            
            # Score debtor keywords
            for keyword, weight in debtor_keywords.items():
                if keyword in text_lower:
                    debtor_score += weight
                    logger.info(f"[ROLE_ASSIGNMENT] {speaker}: Found debtor keyword '{keyword}' (+{weight})")
            
            speaker_scores[speaker] = {
                "collector": collector_score,
                "debtor": debtor_score
            }
            
            logger.info(f"[ROLE_ASSIGNMENT] {speaker}: Collector={collector_score}, Debtor={debtor_score}")
        
        # Step 3: Assign roles based on scores (ALWAYS assign when 2 speakers)
        speakers = list(speaker_scores.keys())
        speaker1, speaker2 = speakers[0], speakers[1]
        
        score1 = speaker_scores[speaker1]
        score2 = speaker_scores[speaker2]
        
        collector_speaker = None
        debtor_speaker = None
        
        # Compare collector scores
        if score1["collector"] > score2["collector"]:
            collector_speaker = speaker1
            debtor_speaker = speaker2
            logger.info(f"[ROLE_ASSIGNMENT] {speaker1} has higher collector score ({score1['collector']} > {score2['collector']})")
        elif score2["collector"] > score1["collector"]:
            collector_speaker = speaker2
            debtor_speaker = speaker1
            logger.info(f"[ROLE_ASSIGNMENT] {speaker2} has higher collector score ({score2['collector']} > {score1['collector']})")
        else:
            # Tied collector scores - check debtor scores
            if score1["debtor"] > score2["debtor"]:
                debtor_speaker = speaker1
                collector_speaker = speaker2
                logger.info(f"[ROLE_ASSIGNMENT] Tied collector scores. {speaker1} has higher debtor score ({score1['debtor']} > {score2['debtor']})")
            elif score2["debtor"] > score1["debtor"]:
                debtor_speaker = speaker2
                collector_speaker = speaker1
                logger.info(f"[ROLE_ASSIGNMENT] Tied collector scores. {speaker2} has higher debtor score ({score2['debtor']} > {score1['debtor']})")
            else:
                # Complete tie - use deterministic fallback: first speaker = COLLECTOR
                first_speaker = min(speaker_first_appearance.keys(), key=lambda s: speaker_first_appearance[s])
                collector_speaker = first_speaker
                debtor_speaker = speaker2 if first_speaker == speaker1 else speaker1
                logger.info(f"[ROLE_ASSIGNMENT] All scores tied. Deterministic fallback: first speaker ({collector_speaker}) -> COLLECTOR")
        
        # ALWAYS log final mapping (guaranteed to have values)
        logger.info(f"[ROLE_ASSIGNMENT] {collector_speaker} -> COLLECTOR")
        logger.info(f"[ROLE_ASSIGNMENT] {debtor_speaker} -> DEBTOR")
        
        # Step 4: Apply roles to turns (preserve original speaker labels)
        for turn in turns:
            speaker = turn.get("speaker", "")
            turn["speaker_id"] = speaker  # Preserve original
            
            if speaker == collector_speaker:
                turn["role"] = "COLLECTOR"
            elif speaker == debtor_speaker:
                turn["role"] = "DEBTOR"
            else:
                turn["role"] = None
        
        # Log summary
        collector_count = sum(1 for t in turns if t.get("role") == "COLLECTOR")
        debtor_count = sum(1 for t in turns if t.get("role") == "DEBTOR")
        
        logger.info(f"[ROLE_ASSIGNMENT] Complete: {collector_count} COLLECTOR turns, {debtor_count} DEBTOR turns")
        
        return turns
