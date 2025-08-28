import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List, Dict
import logging

class FaceDetectionProcessor:
    """Face detection and ROI extraction for rPPG processing"""
    
    def __init__(self, confidence_threshold=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
        
    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face in frame and return bounding box coordinates
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary with face detection results or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]  # Use first detection
                
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                return {
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'landmarks': self._extract_landmarks(detection, w, h)
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            return None
    
    def _extract_landmarks(self, detection, width: int, height: int) -> Dict:
        """Extract facial landmarks from detection"""
        landmarks = {}
        
        if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_keypoints'):
            keypoints = detection.location_data.relative_keypoints
            
            landmark_names = ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 'right_ear', 'left_ear']
            
            for i, keypoint in enumerate(keypoints):
                if i < len(landmark_names):
                    landmarks[landmark_names[i]] = {
                        'x': int(keypoint.x * width),
                        'y': int(keypoint.y * height)
                    }
        
        return landmarks
    
    def extract_roi(self, frame: np.ndarray, roi_type: str = 'forehead') -> Optional[np.ndarray]:
        """
        Extract region of interest for rPPG processing
        
        Args:
            frame: Input frame
            roi_type: Type of ROI ('forehead', 'cheeks', 'full_face')
            
        Returns:
            Extracted ROI as numpy array or None if extraction fails
        """
        face_info = self.detect_face(frame)
        
        if face_info is None:
            return None
        
        x, y, w, h = face_info['bbox']
        
        if roi_type == 'forehead':
            roi_y = y
            roi_h = h // 3
            roi_x = x + w // 4
            roi_w = w // 2
            
        elif roi_type == 'cheeks':
            roi_y = y + h // 3
            roi_h = h // 3
            roi_x = x
            roi_w = w
            
        elif roi_type == 'full_face':
            roi_x, roi_y, roi_w, roi_h = x, y, w, h
            
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")
        
        frame_h, frame_w = frame.shape[:2]
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, frame_w - roi_x)
        roi_h = min(roi_h, frame_h - roi_y)
        
        if roi_w <= 0 or roi_h <= 0:
            return None
        
        return frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    def extract_multiple_rois(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract multiple ROIs for comprehensive rPPG analysis"""
        rois = {}
        
        for roi_type in ['forehead', 'cheeks', 'full_face']:
            roi = self.extract_roi(frame, roi_type)
            if roi is not None:
                rois[roi_type] = roi
        
        return rois
    
    def visualize_detection(self, frame: np.ndarray, show_roi: bool = True) -> np.ndarray:
        """
        Visualize face detection and ROI on frame
        
        Args:
            frame: Input frame
            show_roi: Whether to show ROI rectangles
            
        Returns:
            Frame with visualization overlays
        """
        vis_frame = frame.copy()
        face_info = self.detect_face(frame)
        
        if face_info is None:
            return vis_frame
        
        x, y, w, h = face_info['bbox']
        confidence = face_info['confidence']
        
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_frame, f'Face: {confidence:.2f}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if show_roi:
            roi_y = y
            roi_h = h // 3
            roi_x = x + w // 4
            roi_w = w // 2
            cv2.rectangle(vis_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 1)
            cv2.putText(vis_frame, 'Forehead ROI', (roi_x, roi_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        landmarks = face_info['landmarks']
        for name, point in landmarks.items():
            cv2.circle(vis_frame, (point['x'], point['y']), 2, (0, 0, 255), -1)
        
        return vis_frame
    
    def is_face_stable(self, current_bbox: Tuple[int, int, int, int], 
                      previous_bbox: Tuple[int, int, int, int], 
                      threshold: float = 0.1) -> bool:
        """
        Check if face position is stable between frames
        
        Args:
            current_bbox: Current frame bounding box (x, y, w, h)
            previous_bbox: Previous frame bounding box (x, y, w, h)
            threshold: Stability threshold (relative to face size)
            
        Returns:
            True if face is stable, False otherwise
        """
        if previous_bbox is None:
            return False
        
        curr_x, curr_y, curr_w, curr_h = current_bbox
        prev_x, prev_y, prev_w, prev_h = previous_bbox
        
        dx = abs(curr_x - prev_x) / curr_w
        dy = abs(curr_y - prev_y) / curr_h
        dw = abs(curr_w - prev_w) / curr_w
        dh = abs(curr_h - prev_h) / curr_h
        
        return max(dx, dy, dw, dh) < threshold
