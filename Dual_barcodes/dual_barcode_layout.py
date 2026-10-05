"""
Dual Barcode Layout Generator for Spatial Transcriptomics Arrays

This module implements a block-wise algorithm to assign barcodes to a grid
where each 4x4 block shares the same bar_1 barcode, but has 16 different
bar_2 barcodes to minimize cross-talk while maintaining synthesis accuracy.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
import itertools


class DualBarcodeLayout:
    """
    Generate dual barcode layouts for spatial transcriptomics arrays.
    
    Attributes:
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        barcodes (List[str]): List of available barcodes (used for both bar_1 and bar_2)
        grid_bar1 (np.ndarray): Grid storing bar_1 assignments
        grid_bar2 (np.ndarray): Grid storing bar_2 assignments
        grid_combined (np.ndarray): Grid storing combined bar_1+bar_2 barcodes
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        barcodes: List[str],
        barcodes_bar2: Optional[List[str]] = None,
    ):
        """
        Initialize the dual barcode layout generator.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            barcodes: List of bar_1 barcode sequences (minimum 1 required)
            barcodes_bar2: Optional list of bar_2 barcode sequences
                (minimum 16 required for 4x4 blocks). If omitted, `barcodes`
                is reused for bar_2.
        
        Raises:
            ValueError: If fewer than 16 barcodes provided
        """
        if len(barcodes) < 1:
            raise ValueError("At least one bar_1 barcode is required")

        if barcodes_bar2 is None:
            barcodes_bar2 = barcodes

        if len(barcodes_bar2) < 16:
            raise ValueError("Minimum 16 bar_2 barcodes required to fill any 4x4 block")
        
        self.rows = rows
        self.cols = cols
        self.barcodes = barcodes
        self.barcodes_bar2 = barcodes_bar2
        self.n_barcodes = len(barcodes)
        self.n_barcodes_bar2 = len(barcodes_bar2)
        
        # Initialize grids
        self.grid_bar1 = np.empty((rows, cols), dtype=object)
        self.grid_bar2 = np.empty((rows, cols), dtype=object)
        self.grid_combined = np.empty((rows, cols), dtype=object)
        
        # Calculate number of 4x4 blocks
        self.n_blocks_row = (rows + 3) // 4
        self.n_blocks_col = (cols + 3) // 4
        self.total_blocks = self.n_blocks_row * self.n_blocks_col
        
    def generate_layout(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the complete dual barcode layout.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (grid_bar1, grid_bar2, grid_combined)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Step 1: Assign bar_1 to each 4x4 block
        self._assign_bar1()
        
        # Step 2: Assign bar_2 to positions within each block
        self._assign_bar2()
        
        # Step 3: Create combined barcodes
        self._create_combined_barcodes()
        
        # Step 4: Validate the layout
        self._validate_layout()
        
        return self.grid_bar1, self.grid_bar2, self.grid_combined
    
    def _assign_bar1(self):
        """
        Assign bar_1 barcodes to the grid.
        Each 4x4 block gets the same bar_1 barcode.
        
        Strategy: Distribute bar_1 values to minimize reuse and maximize
        available bar_1+bar_2 combinations across the grid.
        """
        # Calculate how many times each bar_1 will be used
        # With N barcodes, each can be used at most N times (since we have N bar_2 options)
        max_uses_per_barcode = self.n_barcodes_bar2
        
        # Check if we have enough capacity
        if self.total_blocks > self.n_barcodes * max_uses_per_barcode:
            raise ValueError(
                f"Grid too large: {self.total_blocks} blocks require "
                f"{self.total_blocks * 16} spots, but only {self.n_barcodes * max_uses_per_barcode * 16} "
                f"unique combinations possible with {self.n_barcodes} bar_1 and "
                f"{self.n_barcodes_bar2} bar_2 barcodes. "
                f"Increase number of barcodes or reduce grid size."
            )
        
        # Shuffle barcodes for random assignment
        shuffled_barcodes = self.barcodes.copy()
        random.shuffle(shuffled_barcodes)
        
        # Track how many times each barcode has been used
        barcode_usage = {bc: 0 for bc in self.barcodes}
        
        # Assign bar_1 to each block, trying to balance usage
        for block_row in range(self.n_blocks_row):
            for block_col in range(self.n_blocks_col):
                # Find the least-used barcode
                available = [bc for bc in shuffled_barcodes 
                           if barcode_usage[bc] < max_uses_per_barcode]
                
                if not available:
                    raise ValueError("No available bar_1 barcodes (should not happen)")
                
                # Select the least-used barcode
                bar1 = min(available, key=lambda bc: barcode_usage[bc])
                barcode_usage[bar1] += 1
                
                # Assign to all 16 positions in the 4x4 block
                for i in range(4):
                    for j in range(4):
                        row = block_row * 4 + i
                        col = block_col * 4 + j
                        
                        # Check if position is within grid bounds
                        if row < self.rows and col < self.cols:
                            self.grid_bar1[row, col] = bar1
    
    def _assign_bar2(self):
        """
        Assign bar_2 barcodes to the grid.
        Each 4x4 block must have 16 different bar_2 barcodes.
        Additionally, no bar_1+bar_2 combination should repeat across the entire grid.
        
        Strategy: For each block, track which bar_2 values are still available
        for the given bar_1, ensuring global uniqueness.
        """
        # Track which bar_2 values have been used with each bar_1
        bar1_to_used_bar2 = {bc: set() for bc in self.barcodes}
        
        for block_row in range(self.n_blocks_row):
            for block_col in range(self.n_blocks_col):
                # Get all positions in this block that are within grid bounds
                positions = []
                bar1_for_block = None
                
                for i in range(4):
                    for j in range(4):
                        row = block_row * 4 + i
                        col = block_col * 4 + j
                        if row < self.rows and col < self.cols:
                            positions.append((row, col))
                            if bar1_for_block is None:
                                bar1_for_block = self.grid_bar1[row, col]
                
                # Find available bar_2 barcodes for this bar_1
                n_positions = len(positions)
                used_bar2_for_this_bar1 = bar1_to_used_bar2[bar1_for_block]
                available_bar2 = [bc for bc in self.barcodes_bar2
                                 if bc not in used_bar2_for_this_bar1]
                
                # Check if we have enough available bar_2 barcodes
                if len(available_bar2) < n_positions:
                    raise ValueError(
                        f"Cannot find enough unique bar_2 barcodes for block ({block_row}, {block_col}). "
                        f"bar_1='{bar1_for_block}' needs {n_positions} bar_2 values, "
                        f"but only {len(available_bar2)} available: {available_bar2}. "
                        f"Already used with this bar_1: {used_bar2_for_this_bar1}. "
                        f"Try using more barcodes or a smaller grid."
                    )
                
                # Randomly select bar_2 barcodes from available ones
                random.shuffle(available_bar2)
                selected_bar2 = available_bar2[:n_positions]
                
                # Mark these bar_2 values as used with this bar_1
                for bar2 in selected_bar2:
                    used_bar2_for_this_bar1.add(bar2)
                
                # Assign bar_2 to each position
                for (row, col), bar2 in zip(positions, selected_bar2):
                    self.grid_bar2[row, col] = bar2
    
    def _create_combined_barcodes(self):
        """
        Create combined barcodes by concatenating bar_1 and bar_2.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid_bar1[i, j] is not None and self.grid_bar2[i, j] is not None:
                    self.grid_combined[i, j] = self.grid_bar1[i, j] + self.grid_bar2[i, j]
    
    def _validate_layout(self) -> bool:
        """
        Validate that the layout meets all constraints.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Track all combined barcodes globally
        all_combined = {}
        
        # Check each 4x4 block
        for block_row in range(self.n_blocks_row):
            for block_col in range(self.n_blocks_col):
                block_bar1 = set()
                block_bar2 = set()
                block_combined = set()
                
                for i in range(4):
                    for j in range(4):
                        row = block_row * 4 + i
                        col = block_col * 4 + j
                        
                        if row < self.rows and col < self.cols:
                            block_bar1.add(self.grid_bar1[row, col])
                            block_bar2.add(self.grid_bar2[row, col])
                            combined = self.grid_combined[row, col]
                            block_combined.add(combined)
                            
                            # Track global uniqueness
                            if combined in all_combined:
                                raise ValueError(
                                    f"Duplicate combined barcode '{combined}' found at "
                                    f"position ({row}, {col}) and {all_combined[combined]}"
                                )
                            all_combined[combined] = (row, col)
                
                # Validate: all positions in block should have same bar_1
                if len(block_bar1) != 1:
                    raise ValueError(f"Block ({block_row}, {block_col}) has inconsistent bar_1 assignments")
                
                # Validate: all positions in block should have different bar_2
                n_positions = len([1 for i in range(4) for j in range(4) 
                                  if (block_row * 4 + i) < self.rows and (block_col * 4 + j) < self.cols])
                if len(block_bar2) != n_positions:
                    raise ValueError(f"Block ({block_row}, {block_col}) has duplicate bar_2 assignments")
                
                # Validate: all combined barcodes should be unique within block
                if len(block_combined) != n_positions:
                    raise ValueError(f"Block ({block_row}, {block_col}) has duplicate combined barcodes")
        
        return True
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the generated layout.
        
        Returns:
            Dictionary with layout statistics
        """
        unique_combined = len(set(self.grid_combined.flatten()))
        total_spots = self.rows * self.cols
        
        # Count bar_1 usage
        bar1_counts = {}
        for barcode in self.grid_bar1.flatten():
            if barcode is not None:
                bar1_counts[barcode] = bar1_counts.get(barcode, 0) + 1
        
        # Count bar_2 usage
        bar2_counts = {}
        for barcode in self.grid_bar2.flatten():
            if barcode is not None:
                bar2_counts[barcode] = bar2_counts.get(barcode, 0) + 1
        
        return {
            'grid_dimensions': (self.rows, self.cols),
            'total_spots': total_spots,
            'n_barcodes_bar1_available': self.n_barcodes,
            'n_barcodes_bar2_available': self.n_barcodes_bar2,
            'n_blocks': self.total_blocks,
            'unique_combined_barcodes': unique_combined,
            'bar1_usage': bar1_counts,
            'bar2_usage': bar2_counts,
            'theoretical_max_combinations': self.n_barcodes * self.n_barcodes_bar2
        }
    
    def export_layout(self, filepath: str, format: str = 'csv'):
        """
        Export the layout to a file.
        
        Args:
            filepath: Path to save the file
            format: Export format ('csv', 'tsv', or 'npy')
        """
        if format == 'csv':
            self._export_csv(filepath, delimiter=',')
        elif format == 'tsv':
            self._export_csv(filepath, delimiter='\t')
        elif format == 'npy':
            np.save(filepath, {
                'bar1': self.grid_bar1,
                'bar2': self.grid_bar2,
                'combined': self.grid_combined
            })
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_csv(self, filepath: str, delimiter: str = ','):
        """Export layout to CSV/TSV format in column-major order."""
        with open(filepath, 'w') as f:
            # Header
            f.write(f"row{delimiter}col{delimiter}bar_1{delimiter}bar_2{delimiter}combined\n")
            
            # Data rows in column-major order (iterate columns first, then rows)
            for j in range(self.cols):
                for i in range(self.rows):
                    f.write(f"{i}{delimiter}{j}{delimiter}")
                    f.write(f"{self.grid_bar1[i, j]}{delimiter}")
                    f.write(f"{self.grid_bar2[i, j]}{delimiter}")
                    f.write(f"{self.grid_combined[i, j]}\n")
    
    def visualize_layout(self, output_file: Optional[str] = None, include_combined: bool = True):
        """
        Create a visualization of the barcode layout.
        
        Args:
            output_file: Optional path to save the figure
            include_combined: If True, show all 3 plots; if False, only bar_1 and bar_2
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return
        
        # Create color maps for barcodes
        unique_bar1 = list(set(self.grid_bar1.flatten()))
        unique_bar2 = list(set(self.grid_bar2.flatten()))
        
        bar1_colors = {bc: i for i, bc in enumerate(unique_bar1)}
        bar2_colors = {bc: i for i, bc in enumerate(unique_bar2)}
        
        if include_combined:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax_bar1, ax_bar2, ax_combined = axes[0], axes[1], axes[2]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax_bar1, ax_bar2 = axes[0], axes[1]
        
        # Plot bar_1
        bar1_numeric = np.array([[bar1_colors[self.grid_bar1[i, j]] 
                                  for j in range(self.cols)] 
                                 for i in range(self.rows)])
        im1 = ax_bar1.imshow(bar1_numeric, cmap='tab20', aspect='auto')
        ax_bar1.set_title('bar_1 Layout\n(Same within 4x4 blocks)', fontsize=12, fontweight='bold')
        ax_bar1.set_xlabel('Column')
        ax_bar1.set_ylabel('Row')
        
        # Draw block boundaries for bar_1
        for i in range(0, self.rows, 4):
            ax_bar1.axhline(i - 0.5, color='black', linewidth=2)
        for j in range(0, self.cols, 4):
            ax_bar1.axvline(j - 0.5, color='black', linewidth=2)
        
        # Plot bar_2
        bar2_numeric = np.array([[bar2_colors[self.grid_bar2[i, j]] 
                                  for j in range(self.cols)] 
                                 for i in range(self.rows)])
        im2 = ax_bar2.imshow(bar2_numeric, cmap='tab20', aspect='auto')
        ax_bar2.set_title('bar_2 Layout\n(Different within 4x4 blocks)', fontsize=12, fontweight='bold')
        ax_bar2.set_xlabel('Column')
        ax_bar2.set_ylabel('Row')
        
        # Draw block boundaries for bar_2
        for i in range(0, self.rows, 4):
            ax_bar2.axhline(i - 0.5, color='black', linewidth=2)
        for j in range(0, self.cols, 4):
            ax_bar2.axvline(j - 0.5, color='black', linewidth=2)
        
        # Plot combined with text annotations (only if requested)
        if include_combined:
            ax_combined.set_xlim(-0.5, self.cols - 0.5)
            ax_combined.set_ylim(self.rows - 0.5, -0.5)
            ax_combined.set_aspect('equal')
            
            for i in range(self.rows):
                for j in range(self.cols):
                    combined = self.grid_combined[i, j]
                    # Color code by bar_1
                    color_idx = bar1_colors[self.grid_bar1[i, j]]
                    color = plt.cm.tab20(color_idx / len(unique_bar1))
                    
                    rect = mpatches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, 
                                              facecolor=color, edgecolor='black', linewidth=0.5)
                    ax_combined.add_patch(rect)
                    
                    # Add text
                    fontsize = max(6, min(10, 100 // max(self.rows, self.cols)))
                    ax_combined.text(j, i, combined, ha='center', va='center', 
                               fontsize=fontsize, fontweight='bold')
            
            # Draw block boundaries
            for i in range(0, self.rows, 4):
                ax_combined.axhline(i - 0.5, color='red', linewidth=2)
            for j in range(0, self.cols, 4):
                ax_combined.axvline(j - 0.5, color='red', linewidth=2)
            
            ax_combined.set_title('Combined Layout (bar_1 + bar_2)', fontsize=12, fontweight='bold')
            ax_combined.set_xlabel('Column')
            ax_combined.set_ylabel('Row')
            ax_combined.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()


def generate_random_barcodes(n: int, length: int = 8, seed: Optional[int] = None) -> List[str]:
    """
    Generate random DNA barcodes.
    
    Args:
        n: Number of barcodes to generate
        length: Length of each barcode
        seed: Random seed for reproducibility
    
    Returns:
        List of random DNA barcode sequences
    """
    if seed is not None:
        random.seed(seed)
    
    bases = ['A', 'C', 'G', 'T']
    barcodes = []
    
    for _ in range(n):
        barcode = ''.join(random.choices(bases, k=length))
        barcodes.append(barcode)
    
    return barcodes


def load_barcodes_from_file(filepath: str) -> List[str]:
    """
    Load barcodes from a text file (one barcode per line).
    
    Args:
        filepath: Path to the text file containing barcodes
    
    Returns:
        List of barcode sequences
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or contains invalid barcodes
    """
    try:
        with open(filepath, 'r') as f:
            barcodes = [line.strip() for line in f if line.strip()]
        
        if not barcodes:
            raise ValueError(f"No barcodes found in file: {filepath}")
        
        # Validate that all barcodes are non-empty strings
        for i, bc in enumerate(barcodes):
            if not bc:
                raise ValueError(f"Empty barcode at line {i+1}")
        
        print(f"Loaded {len(barcodes)} barcodes from {filepath}")
        return barcodes
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Barcode file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading barcode file: {e}")


if __name__ == "__main__":
    import sys
    import os
    
    # Example usage
    print("=" * 70)
    print("Dual Barcode Layout Generator for Spatial Transcriptomics")
    print("=" * 70)
    
    # Parameters
    ROWS = 8
    COLS = 8
    SEED = 42
    BARCODE_FILE = None  # Set to a file path to load barcodes from file
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        BARCODE_FILE = sys.argv[1]
        if len(sys.argv) > 2:
            ROWS = int(sys.argv[2])
        if len(sys.argv) > 3:
            COLS = int(sys.argv[3])
    
    # Calculate grid requirements
    N_BLOCKS = ((ROWS + 3) // 4) * ((COLS + 3) // 4)  # Number of 4x4 blocks
    
    print(f"\nGrid size: {ROWS}x{COLS} ({ROWS*COLS} spots)")
    print(f"Number of 4x4 blocks: {N_BLOCKS}")
    print(f"Minimum barcodes needed: {N_BLOCKS}")
    
    # Load or generate barcodes
    if BARCODE_FILE and os.path.exists(BARCODE_FILE):
        print(f"\nLoading barcodes from file: {BARCODE_FILE}")
        barcodes = load_barcodes_from_file(BARCODE_FILE)
        print(f"First 5 barcodes: {barcodes[:5] if len(barcodes) > 5 else barcodes}")
    else:
        # Generate random barcodes as fallback
        N_BARCODES = N_BLOCKS  # Need at least as many barcodes as blocks
        BARCODE_LENGTH = 6
        if BARCODE_FILE:
            print(f"\n⚠ Warning: Barcode file '{BARCODE_FILE}' not found. Generating random barcodes instead.")
        print(f"\nGenerating {N_BARCODES} random barcodes of length {BARCODE_LENGTH}...")
        barcodes = generate_random_barcodes(N_BARCODES, BARCODE_LENGTH, seed=SEED)
        print(f"First 5 barcodes: {barcodes[:5] if len(barcodes) > 5 else barcodes}")
    
    # Validate barcode count
    if len(barcodes) < N_BLOCKS:
        print(f"\n⚠ WARNING: Only {len(barcodes)} barcodes available, but {N_BLOCKS} blocks require at least {N_BLOCKS} barcodes.")
        print(f"  Each barcode can only be used as bar_1 once (needs 16 unique bar_2 per block).")
        print(f"  Consider reducing grid size or adding more barcodes.")
        sys.exit(1)
    
    # Create layout
    print(f"\nCreating {ROWS}x{COLS} grid layout...")
    layout = DualBarcodeLayout(ROWS, COLS, barcodes)
    
    # Generate the layout
    try:
        grid_bar1, grid_bar2, grid_combined = layout.generate_layout(seed=SEED)
        print("✓ Layout generated successfully!")
        
        # Get statistics
        stats = layout.get_statistics()
        print("\n" + "=" * 70)
        print("LAYOUT STATISTICS")
        print("=" * 70)
        print(f"Grid dimensions: {stats['grid_dimensions']}")
        print(f"Total spots: {stats['total_spots']}")
        print(f"Number of barcodes available: {stats['n_barcodes_available']}")
        print(f"Number of 4x4 blocks: {stats['n_blocks']}")
        print(f"Unique combined barcodes: {stats['unique_combined_barcodes']}")
        print(f"Theoretical max combinations: {stats['theoretical_max_combinations']}")
        
        print("\nbar_1 usage distribution:")
        for barcode, count in sorted(stats['bar1_usage'].items()):
            print(f"  {barcode}: {count} spots")
        
        print("\nbar_2 usage distribution:")
        for barcode, count in sorted(stats['bar2_usage'].items()):
            print(f"  {barcode}: {count} spots")
        
        # Export layout
        print("\n" + "=" * 70)
        print("EXPORTING LAYOUTS")
        print("=" * 70)
        layout.export_layout('barcode_layout.csv', format='csv')
        print("✓ Layout exported to barcode_layout.csv")
        
        # Visualize (optional - requires matplotlib)
        print("\nGenerating visualization...")
        layout.visualize_layout('barcode_layout_visualization.png')
        
        print("\n" + "=" * 70)
        print("DONE!")
        print("=" * 70)
        print("\nUsage:")
        print("  python dual_barcode_layout.py [barcode_file] [rows] [cols]")
        print("  Example: python dual_barcode_layout.py barcodes.txt 8 8")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
