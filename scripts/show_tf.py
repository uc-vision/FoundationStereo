#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
from typing import Dict, Set, List
from beartype import beartype


class TFFramePrinter(Node):
    """Prints TF frame tree in human-readable format."""
    
    def __init__(self) -> None:
        super().__init__('tf_frame_printer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
    @beartype
    def get_frame_relationships(self) -> Dict[str, Set[str]]:
        """Extract parent->children relationships from TF buffer."""
        # Wait for some transforms to populate and spin to receive messages
        import rclpy
        for i in range(30):  # Spin for 3 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        
        relationships: Dict[str, Set[str]] = {}
        all_frames_string = self.tf_buffer.all_frames_as_string()
        all_frames = all_frames_string.split('\n')
        
        for line in all_frames:
            if 'Frame' in line and 'exists with parent' in line:
                # Parse: "Frame camera_link exists with parent base_link."
                parts = line.strip().split()
                if len(parts) >= 6:
                    child = parts[1]
                    parent = parts[5].rstrip('.')
                    
                    if parent not in relationships:
                        relationships[parent] = set()
                    relationships[parent].add(child)
        
        return relationships
    
    @beartype
    def find_root_frames(self, relationships: Dict[str, Set[str]]) -> List[str]:
        """Find frames that have no parents (root nodes)."""
        all_children = set()
        for children in relationships.values():
            all_children.update(children)
        
        all_parents = set(relationships.keys())
        return list(all_parents - all_children)
    
    @beartype
    def print_tree(self, frame: str, relationships: Dict[str, Set[str]], 
                   prefix: str = "", is_last: bool = True) -> None:
        """Recursively print the frame tree."""
        # Choose the appropriate tree characters
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{frame}")
        
        # Get children and sort them for consistent output
        children = sorted(relationships.get(frame, set()))
        
        # Prepare prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Print each child
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            self.print_tree(child, relationships, child_prefix, is_last_child)
    
    @beartype
    def print_all_frames(self) -> None:
        """Print all TF frames in a tree structure."""
        relationships = self.get_frame_relationships()
        
        if not relationships:
            print("No TF frames found. Make sure transforms are being published.")
            return
        
        root_frames = self.find_root_frames(relationships)
        
        print("TF Frame Tree:")
        print("=" * 40)
        
        for i, root in enumerate(sorted(root_frames)):
            if i > 0:
                print()  # Add spacing between separate trees
            self.print_tree(root, relationships)
        
        # Also show any orphaned frames (shouldn't happen in well-formed trees)
        all_mentioned = set(relationships.keys())
        for children in relationships.values():
            all_mentioned.update(children)
        
        orphans = []
        for parent in relationships:
            if parent not in {child for children in relationships.values() for child in children}:
                continue
        
        print(f"\nTotal frames: {len(all_mentioned)}")


def main():
    rclpy.init()
    
    try:
        printer = TFFramePrinter()
        printer.print_all_frames()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()