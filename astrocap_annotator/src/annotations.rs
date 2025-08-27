use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root structure for manual annotations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnnotationSession {
    /// Metadata about the video being annotated
    pub video_metadata: VideoMetadata,
    /// List of tracked objects across the video
    pub objects: Vec<TrackedObject>,
    /// Annotation session metadata
    pub session_metadata: AnnotationSessionMetadata,
}

/// Video file information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VideoMetadata {
    pub file_path: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub duration_seconds: f32,
}

/// Represents a single tracked object (star) across multiple frames
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrackedObject {
    /// Unique identifier for this object
    pub id: u32,
    /// Human-readable name/description
    pub name: String,
    /// Object type (e.g., "star", "planet", "satellite")
    pub object_type: ObjectType,
    /// Keyframe positions that define the track
    pub keyframes: Vec<Keyframe>,
    /// Optional metadata for this object
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of objects that can be tracked
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ObjectType {
    Star,
    Planet,
    Satellite,
    Asteroid,
    Other(String),
}

/// A keyframe position in the track
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Keyframe {
    /// Frame number (0-based)
    pub frame_number: u32,
    /// Sub-pixel X coordinate
    pub x: f32,
    /// Sub-pixel Y coordinate
    pub y: f32,
    /// Confidence in this annotation (0.0 - 1.0)
    pub confidence: f32,
    /// Optional notes about this keyframe
    pub notes: Option<String>,
}

/// Information about the annotation session
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnnotationSessionMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub annotator: String,
    pub tool_version: String,
    pub notes: Option<String>,
}

impl TrackedObject {
    /// Get the position of this object at a specific frame using linear interpolation
    pub fn position_at_frame(&self, frame_number: u32) -> Option<(f32, f32)> {
        // Find exact keyframe match first
        if let Some(keyframe) = self
            .keyframes
            .iter()
            .find(|k| k.frame_number == frame_number)
        {
            return Some((keyframe.x, keyframe.y));
        }

        // Find bounding keyframes for interpolation
        let mut before: Option<&Keyframe> = None;
        let mut after: Option<&Keyframe> = None;

        for keyframe in &self.keyframes {
            if keyframe.frame_number < frame_number {
                if before.is_none() || keyframe.frame_number > before.unwrap().frame_number {
                    before = Some(keyframe);
                }
            } else if keyframe.frame_number > frame_number {
                if after.is_none() || keyframe.frame_number < after.unwrap().frame_number {
                    after = Some(keyframe);
                }
            }
        }

        // Linear interpolation between keyframes
        match (before, after) {
            (Some(b), Some(a)) => {
                let t = (frame_number - b.frame_number) as f32
                    / (a.frame_number - b.frame_number) as f32;
                let x = b.x + t * (a.x - b.x);
                let y = b.y + t * (a.y - b.y);
                Some((x, y))
            }
            _ => None, // No interpolation possible
        }
    }

    /// Get the frame range where this object is visible
    pub fn frame_range(&self) -> Option<(u32, u32)> {
        if self.keyframes.is_empty() {
            return None;
        }

        let min_frame = self.keyframes.iter().map(|k| k.frame_number).min().unwrap();
        let max_frame = self.keyframes.iter().map(|k| k.frame_number).max().unwrap();
        Some((min_frame, max_frame))
    }

    /// Add a new keyframe to this track
    pub fn add_keyframe(&mut self, keyframe: Keyframe) {
        // Insert in sorted order by frame number
        match self
            .keyframes
            .binary_search_by_key(&keyframe.frame_number, |k| k.frame_number)
        {
            Ok(pos) => {
                // Replace existing keyframe at this frame
                self.keyframes[pos] = keyframe;
            }
            Err(pos) => {
                // Insert at the correct position
                self.keyframes.insert(pos, keyframe);
            }
        }
    }
}

/// Utility functions for working with annotations
impl AnnotationSession {
    pub fn new(video_path: String, width: u32, height: u32, frame_count: u32, fps: f32) -> Self {
        Self {
            video_metadata: VideoMetadata {
                file_path: video_path,
                width,
                height,
                frame_count,
                fps,
                duration_seconds: frame_count as f32 / fps,
            },
            objects: Vec::new(),
            session_metadata: AnnotationSessionMetadata {
                created_at: chrono::Utc::now(),
                annotator: "manual".to_string(),
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
                notes: None,
            },
        }
    }

    /// Get all objects visible at a specific frame
    pub fn objects_at_frame(&self, frame_number: u32) -> Vec<(u32, f32, f32)> {
        self.objects
            .iter()
            .filter_map(|obj| {
                obj.position_at_frame(frame_number)
                    .map(|(x, y)| (obj.id, x, y))
            })
            .collect()
    }

    /// Add a new tracked object
    pub fn add_object(&mut self, name: String, object_type: ObjectType) -> u32 {
        let id = self.objects.iter().map(|o| o.id).max().unwrap_or(0) + 1;
        self.objects.push(TrackedObject {
            id,
            name,
            object_type,
            keyframes: Vec::new(),
            metadata: HashMap::new(),
        });
        id
    }

    /// Save annotations to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load annotations from JSON file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let annotations = serde_json::from_str(&json)?;
        Ok(annotations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let obj = TrackedObject {
            id: 1,
            name: "Test Star".to_string(),
            object_type: ObjectType::Star,
            keyframes: vec![
                Keyframe {
                    frame_number: 10,
                    x: 100.0,
                    y: 200.0,
                    confidence: 1.0,
                    notes: None,
                },
                Keyframe {
                    frame_number: 20,
                    x: 200.0,
                    y: 300.0,
                    confidence: 1.0,
                    notes: None,
                },
            ],
            metadata: HashMap::new(),
        };

        // Test exact keyframe match
        assert_eq!(obj.position_at_frame(10), Some((100.0, 200.0)));
        assert_eq!(obj.position_at_frame(20), Some((200.0, 300.0)));

        // Test interpolation at midpoint
        assert_eq!(obj.position_at_frame(15), Some((150.0, 250.0)));

        // Test outside range
        assert_eq!(obj.position_at_frame(5), None);
        assert_eq!(obj.position_at_frame(25), None);
    }

    #[test]
    fn test_keyframe_insertion() {
        let mut obj = TrackedObject {
            id: 1,
            name: "Test".to_string(),
            object_type: ObjectType::Star,
            keyframes: Vec::new(),
            metadata: HashMap::new(),
        };

        // Add keyframes out of order
        obj.add_keyframe(Keyframe {
            frame_number: 20,
            x: 20.0,
            y: 20.0,
            confidence: 1.0,
            notes: None,
        });

        obj.add_keyframe(Keyframe {
            frame_number: 10,
            x: 10.0,
            y: 10.0,
            confidence: 1.0,
            notes: None,
        });

        obj.add_keyframe(Keyframe {
            frame_number: 15,
            x: 15.0,
            y: 15.0,
            confidence: 1.0,
            notes: None,
        });

        // Should be sorted by frame number
        assert_eq!(obj.keyframes[0].frame_number, 10);
        assert_eq!(obj.keyframes[1].frame_number, 15);
        assert_eq!(obj.keyframes[2].frame_number, 20);

        // Test interpolation works with sorted keyframes
        assert_eq!(obj.position_at_frame(12), Some((12.0, 12.0)));
    }
}
