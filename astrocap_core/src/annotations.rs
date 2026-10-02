use std::collections::HashMap;

use serde::{Deserialize, Serialize};
pub use vyd::annotations::{AnnotationSessionMetadata, Keyframe, VideoMetadata};

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
    Unknown,
    Star,
    Planet,
    Satellite,
    Asteroid,
    Other(String),
}

impl TrackedObject {
    pub fn position_at_frame(&self, frame_number: u32) -> Option<(f32, f32)> {
        vyd::annotations::position_at_frame(&self.keyframes, frame_number)
    }

    pub fn frame_range(&self) -> Option<(u32, u32)> {
        vyd::annotations::frame_range(&self.keyframes)
    }

    pub fn add_keyframe(&mut self, keyframe: Keyframe) {
        vyd::annotations::add_keyframe(&mut self.keyframes, keyframe);
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
