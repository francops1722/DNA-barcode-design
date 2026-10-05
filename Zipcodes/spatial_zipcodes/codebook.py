"""
Codebook generation and export module.

Assembles full spatial zip codes from X and Y segments and exports
to structured formats (CSV/TSV).
"""

import logging
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    """Represents a single spatial feature with its zip code."""
    
    feature_id: str
    row: int
    col: int
    x_bits: str
    y_bits: str
    x_seq: str
    y_seq: str
    zip_seq: str
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for export."""
        return {
            'feature_id': self.feature_id,
            'row': self.row,
            'col': self.col,
            'X_bits': self.x_bits,
            'Y_bits': self.y_bits,
            'X_seq': self.x_seq,
            'Y_seq': self.y_seq,
            'zip_seq': self.zip_seq,
        }


class Codebook:
    """
    Manages spatial zipcode codebook.
    
    Maps each (row, col) position on the chip to its unique zip code.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        x_codes: List[List[int]],
        y_codes: List[List[int]],
        x_sequences: List[str],
        y_sequences: List[str],
        spacer: str = "GGG",
        feature_id_format: str = "row_col",
    ):
        """
        Initialize codebook.
        
        Args:
            rows: Number of rows on chip
            cols: Number of columns on chip
            x_codes: Binary codes for X axis (one per column)
            y_codes: Binary codes for Y axis (one per row)
            x_sequences: DNA sequences for X axis
            y_sequences: DNA sequences for Y axis
            spacer: Spacer sequence between X and Y
            feature_id_format: How to format feature IDs ("row_col" or "linear")
        """
        self.rows = rows
        self.cols = cols
        self.x_codes = x_codes
        self.y_codes = y_codes
        self.x_sequences = x_sequences
        self.y_sequences = y_sequences
        self.spacer = spacer
        self.feature_id_format = feature_id_format
        
        # Validate inputs
        if len(x_codes) != cols:
            raise ValueError(f"Expected {cols} X codes, got {len(x_codes)}")
        if len(y_codes) != rows:
            raise ValueError(f"Expected {rows} Y codes, got {len(y_codes)}")
        if len(x_sequences) != cols:
            raise ValueError(f"Expected {cols} X sequences, got {len(x_sequences)}")
        if len(y_sequences) != rows:
            raise ValueError(f"Expected {rows} Y sequences, got {len(y_sequences)}")
        
        self.features: List[Feature] = []
        self._build_features()
        
        logger.info(
            f"Initialized codebook: {rows}x{cols} = {len(self.features)} features, "
            f"spacer='{spacer}'"
        )
    
    def _format_feature_id(self, row: int, col: int) -> str:
        """Generate feature ID based on format."""
        if self.feature_id_format == "row_col":
            return f"R{row:03d}_C{col:03d}"
        elif self.feature_id_format == "linear":
            return str(row * self.cols + col)
        else:
            # Default to row_col format
            return f"R{row:03d}_C{col:03d}"
    
    def _bits_to_string(self, bits: List[int]) -> str:
        """Convert bit list to string."""
        return ''.join(map(str, bits))
    
    def _build_features(self):
        """Build feature list for all (row, col) positions."""
        self.features = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                feature_id = self._format_feature_id(row, col)
                
                # Get codes and sequences
                x_bits = self.x_codes[col]
                y_bits = self.y_codes[row]
                x_seq = self.x_sequences[col]
                y_seq = self.y_sequences[row]
                
                # Assemble full zip code
                zip_seq = x_seq + self.spacer + y_seq
                
                feature = Feature(
                    feature_id=feature_id,
                    row=row,
                    col=col,
                    x_bits=self._bits_to_string(x_bits),
                    y_bits=self._bits_to_string(y_bits),
                    x_seq=x_seq,
                    y_seq=y_seq,
                    zip_seq=zip_seq,
                )
                
                self.features.append(feature)
    
    def get_feature(self, row: int, col: int) -> Optional[Feature]:
        """Get feature by row and column."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        
        idx = row * self.cols + col
        return self.features[idx]
    
    def get_feature_by_id(self, feature_id: str) -> Optional[Feature]:
        """Get feature by ID."""
        for feature in self.features:
            if feature.feature_id == feature_id:
                return feature
        return None
    
    def get_all_zip_sequences(self) -> List[str]:
        """Get all zip code sequences."""
        return [f.zip_seq for f in self.features]
    
    def export_to_csv(
        self,
        output_path: Path,
        delimiter: str = ',',
        include_header: bool = True,
    ):
        """
        Export codebook to CSV/TSV file.
        
        Args:
            output_path: Path to output file
            delimiter: Field delimiter (',' for CSV, '\\t' for TSV)
            include_header: Whether to include header row
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'feature_id', 'row', 'col', 'X_bits', 'Y_bits',
            'X_seq', 'Y_seq', 'zip_seq'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            
            if include_header:
                writer.writeheader()
            
            for feature in self.features:
                writer.writerow(feature.to_dict())
        
        logger.info(f"Exported codebook to {output_path} ({len(self.features)} features)")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get codebook statistics."""
        zip_lengths = [len(f.zip_seq) for f in self.features]
        
        stats = {
            'n_features': len(self.features),
            'rows': self.rows,
            'cols': self.cols,
            'x_bit_length': len(self.x_codes[0]) if self.x_codes else 0,
            'y_bit_length': len(self.y_codes[0]) if self.y_codes else 0,
            'x_dna_length': len(self.x_sequences[0]) if self.x_sequences else 0,
            'y_dna_length': len(self.y_sequences[0]) if self.y_sequences else 0,
            'spacer': self.spacer,
            'spacer_length': len(self.spacer),
            'zip_length': zip_lengths[0] if zip_lengths else 0,
        }
        
        return stats


def build_codebook(
    rows: int,
    cols: int,
    x_codes: List[List[int]],
    y_codes: List[List[int]],
    x_sequences: List[str],
    y_sequences: List[str],
    spacer: str = "GGG",
    feature_id_format: str = "row_col",
) -> Codebook:
    """
    Build a codebook from X and Y codes and sequences.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        x_codes: Binary codes for X axis
        y_codes: Binary codes for Y axis
        x_sequences: DNA sequences for X axis
        y_sequences: DNA sequences for Y axis
        spacer: Spacer sequence
        feature_id_format: Feature ID format
        
    Returns:
        Codebook object
    """
    return Codebook(
        rows=rows,
        cols=cols,
        x_codes=x_codes,
        y_codes=y_codes,
        x_sequences=x_sequences,
        y_sequences=y_sequences,
        spacer=spacer,
        feature_id_format=feature_id_format,
    )


def export_codebook(
    codebook: Codebook,
    output_path: Path,
    file_format: str = "csv",
):
    """
    Export codebook to file.
    
    Args:
        codebook: Codebook to export
        output_path: Output file path
        file_format: Format ("csv" or "tsv")
    """
    delimiter = '\t' if file_format == 'tsv' else ','
    
    if file_format == 'tsv' and not str(output_path).endswith('.tsv'):
        output_path = Path(str(output_path).replace('.csv', '.tsv'))
    
    codebook.export_to_csv(output_path, delimiter=delimiter)


def load_codebook_from_csv(
    input_path: Path,
    delimiter: str = ',',
) -> List[Feature]:
    """
    Load codebook from CSV/TSV file.
    
    Args:
        input_path: Path to input file
        delimiter: Field delimiter
        
    Returns:
        List of Feature objects
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Codebook file not found: {input_path}")
    
    features = []
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            feature = Feature(
                feature_id=row['feature_id'],
                row=int(row['row']),
                col=int(row['col']),
                x_bits=row['X_bits'],
                y_bits=row['Y_bits'],
                x_seq=row['X_seq'],
                y_seq=row['Y_seq'],
                zip_seq=row['zip_seq'],
            )
            features.append(feature)
    
    logger.info(f"Loaded {len(features)} features from {input_path}")
    return features
