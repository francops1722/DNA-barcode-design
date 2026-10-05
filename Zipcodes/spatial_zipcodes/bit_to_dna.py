"""
Bit-to-DNA mapping module.

Implements deterministic, invertible mappings between binary codes
and DNA sequences, supporting multiple encoding schemes including
photolithography-realistic 4-bit one-hot encoding.
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class BitToDNAEncoder:
    """
    Encodes binary bit strings into DNA sequences using configurable schemes.
    
    Supports:
    - 2-bits-per-base: Each base encodes 2 bits (default: 00->A, 01->C, 10->G, 11->T)
    - 1-bit-per-base: Each base encodes 1 bit (0->A, 1->T)
    - 4-bits-per-base: One-hot encoding for photolithography (1000->A, 0100->C, 0010->G, 0001->T)
    """
    
    # Default mappings
    DEFAULT_2BIT_MAPPING = {
        "00": "A",
        "01": "C",
        "10": "G",
        "11": "T",
    }
    
    DEFAULT_1BIT_MAPPING = {
        "0": "A",
        "1": "T",
    }
    
    DEFAULT_4BIT_MAPPING = {
        "1000": "A",  # One-hot: A synthesis cycle
        "0100": "C",  # One-hot: C synthesis cycle
        "0010": "G",  # One-hot: G synthesis cycle
        "0001": "T",  # One-hot: T synthesis cycle
    }
    
    def __init__(self, scheme: str = "2-bits-per-base", mapping: Dict[str, str] = None):
        """
        Initialize encoder.
        
        Args:
            scheme: Encoding scheme ("2-bits-per-base", "1-bit-per-base", or "4-bits-per-base")
            mapping: Custom bit-to-base mapping (optional)
        """
        self.scheme = scheme
        
        if scheme == "2-bits-per-base":
            self.bits_per_base = 2
            self.mapping = mapping if mapping else self.DEFAULT_2BIT_MAPPING.copy()
        elif scheme == "1-bit-per-base":
            self.bits_per_base = 1
            self.mapping = mapping if mapping else self.DEFAULT_1BIT_MAPPING.copy()
        elif scheme == "4-bits-per-base":
            self.bits_per_base = 4
            self.mapping = mapping if mapping else self.DEFAULT_4BIT_MAPPING.copy()
        else:
            raise ValueError(f"Unknown encoding scheme: {scheme}")
        
        # Create reverse mapping for decoding
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        
        # Validate mapping
        self._validate_mapping()
        
        logger.info(f"Initialized BitToDNAEncoder with scheme '{scheme}'")
    
    def _validate_mapping(self):
        """Validate that mapping is complete and invertible."""
        # For 4-bit one-hot encoding, we only need 4 entries (one per base)
        # For other schemes, we need all possible bit patterns
        if self.scheme == "4-bits-per-base":
            # One-hot encoding: exactly 4 entries, one for each base
            if len(self.mapping) != 4:
                raise ValueError(
                    f"4-bit one-hot mapping must have exactly 4 entries (one per base), "
                    f"got {len(self.mapping)}"
                )
            # Verify one-hot patterns (exactly one bit set)
            for bit_pattern in self.mapping.keys():
                if bit_pattern.count('1') != 1:
                    raise ValueError(
                        f"4-bit one-hot encoding requires exactly one '1' per pattern, "
                        f"got: {bit_pattern}"
                    )
        else:
            # Standard validation for 1-bit and 2-bit schemes
            expected_keys = 2 ** self.bits_per_base
            if len(self.mapping) != expected_keys:
                raise ValueError(
                    f"Mapping must have {expected_keys} entries for {self.bits_per_base}-bit scheme, "
                    f"got {len(self.mapping)}"
                )
            
            # Check all bit patterns are covered
            for i in range(expected_keys):
                bit_pattern = format(i, f'0{self.bits_per_base}b')
                if bit_pattern not in self.mapping:
                    raise ValueError(f"Mapping missing bit pattern: {bit_pattern}")
        
        # Check bases are unique (invertible)
        bases = list(self.mapping.values())
        if len(bases) != len(set(bases)):
            raise ValueError("Mapping contains duplicate bases (not invertible)")
        
        # Check bases are valid DNA
        valid_bases = set("ACGT")
        for base in bases:
            if base not in valid_bases:
                raise ValueError(f"Invalid DNA base in mapping: {base}")
    
    def encode(self, bits: List[int]) -> str:
        """
        Encode binary code into DNA sequence.
        
        For 4-bit one-hot: converts 2-bit patterns to one-hot (00->1000, 01->0100, 10->0010, 11->0001)
        For other schemes: directly maps bit patterns to bases
        
        Args:
            bits: List of 0s and 1s
            
        Returns:
            DNA sequence string
            
        Raises:
            ValueError: If bit length is incompatible with scheme
        """
        if self.scheme == "4-bits-per-base":
            # For one-hot encoding, input should be 2-bit patterns
            # Convert each 2-bit pattern to 4-bit one-hot
            if len(bits) % 2 != 0:
                raise ValueError(
                    f"Bit length ({len(bits)}) must be multiple of 2 for 4-bit one-hot encoding"
                )
            
            # Map 2-bit patterns to one-hot 4-bit
            two_to_onehot = {
                "00": "1000",
                "01": "0100",
                "10": "0010",
                "11": "0001",
            }
            
            dna = []
            bit_string = ''.join(map(str, bits))
            
            for i in range(0, len(bit_string), 2):
                two_bit = bit_string[i:i + 2]
                onehot = two_to_onehot[two_bit]
                if onehot not in self.mapping:
                    raise ValueError(f"One-hot pattern {onehot} not in mapping")
                dna.append(self.mapping[onehot])
            
            return ''.join(dna)
        else:
            # Standard encoding for 1-bit and 2-bit schemes
            if len(bits) % self.bits_per_base != 0:
                raise ValueError(
                    f"Bit length ({len(bits)}) must be multiple of {self.bits_per_base} "
                    f"for {self.scheme} scheme"
                )
            
            # Convert bits to string
            bit_string = ''.join(map(str, bits))
            
            # Encode chunks
            dna = []
            for i in range(0, len(bit_string), self.bits_per_base):
                chunk = bit_string[i:i + self.bits_per_base]
                if chunk not in self.mapping:
                    raise ValueError(f"Invalid bit pattern: {chunk}")
                dna.append(self.mapping[chunk])
            
            return ''.join(dna)
    
    def decode(self, dna: str) -> List[int]:
        """
        Decode DNA sequence back into binary code.
        
        For 4-bit one-hot: converts back to 2-bit patterns (1000->00, 0100->01, 0010->10, 0001->11)
        For other schemes: directly decodes bases to bit patterns
        
        Args:
            dna: DNA sequence string
            
        Returns:
            List of 0s and 1s
            
        Raises:
            ValueError: If sequence contains invalid bases
        """
        if self.scheme == "4-bits-per-base":
            # For one-hot encoding, decode to 2-bit patterns
            onehot_to_two = {
                "1000": "00",
                "0100": "01",
                "0010": "10",
                "0001": "11",
            }
            
            bits = []
            for base in dna:
                if base not in self.reverse_mapping:
                    raise ValueError(f"Invalid or unmapped DNA base: {base}")
                
                onehot = self.reverse_mapping[base]
                two_bit = onehot_to_two[onehot]
                bits.extend(int(b) for b in two_bit)
            
            return bits
        else:
            # Standard decoding for 1-bit and 2-bit schemes
            bits = []
            
            for base in dna:
                if base not in self.reverse_mapping:
                    raise ValueError(f"Invalid or unmapped DNA base: {base}")
                
                bit_pattern = self.reverse_mapping[base]
                bits.extend(int(b) for b in bit_pattern)
            
            return bits
    
    def get_dna_length(self, bit_length: int) -> int:
        """
        Calculate DNA sequence length for given bit length.
        
        For 4-bit one-hot: input is 2-bit patterns, output is 1 base per 2 bits
        For other schemes: standard bits_per_base division
        
        Args:
            bit_length: Number of bits
            
        Returns:
            DNA sequence length in bases
        """
        if self.scheme == "4-bits-per-base":
            # One-hot: 2 input bits -> 1 base
            if bit_length % 2 != 0:
                raise ValueError(f"Bit length ({bit_length}) must be multiple of 2 for one-hot encoding")
            return bit_length // 2
        else:
            if bit_length % self.bits_per_base != 0:
                raise ValueError(
                    f"Bit length ({bit_length}) must be multiple of {self.bits_per_base}"
                )
            return bit_length // self.bits_per_base
    
    def get_bit_length(self, dna_length: int) -> int:
        """
        Calculate bit length for given DNA sequence length.
        
        For 4-bit one-hot: 1 base -> 2 bits (decoded format)
        For other schemes: standard bits_per_base multiplication
        
        Args:
            dna_length: DNA sequence length in bases
            
        Returns:
            Number of bits
        """
        if self.scheme == "4-bits-per-base":
            # One-hot: 1 base -> 2 bits (decoded)
            return dna_length * 2
        else:
            return dna_length * self.bits_per_base


def encode_codes_to_dna(
    codes: List[List[int]],
    scheme: str = "2-bits-per-base",
    mapping: Dict[str, str] = None,
) -> List[str]:
    """
    Encode multiple binary codes to DNA sequences.
    
    Args:
        codes: List of binary codes (lists of 0s and 1s)
        scheme: Encoding scheme
        mapping: Optional custom mapping
        
    Returns:
        List of DNA sequences
    """
    encoder = BitToDNAEncoder(scheme=scheme, mapping=mapping)
    return [encoder.encode(code) for code in codes]


def decode_dna_to_codes(
    sequences: List[str],
    scheme: str = "2-bits-per-base",
    mapping: Dict[str, str] = None,
) -> List[List[int]]:
    """
    Decode multiple DNA sequences to binary codes.
    
    Args:
        sequences: List of DNA sequences
        scheme: Encoding scheme
        mapping: Optional custom mapping
        
    Returns:
        List of binary codes
    """
    encoder = BitToDNAEncoder(scheme=scheme, mapping=mapping)
    return [encoder.decode(seq) for seq in sequences]


def hamming_distance_dna(seq1: str, seq2: str) -> int:
    """
    Calculate Hamming distance between two DNA sequences.
    
    Args:
        seq1: First DNA sequence
        seq2: Second DNA sequence
        
    Returns:
        Hamming distance (number of differing bases)
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must have same length: {len(seq1)} vs {len(seq2)}")
    return sum(b1 != b2 for b1, b2 in zip(seq1, seq2))


def analyze_encoding_distance_preservation(
    codes: List[List[int]],
    scheme: str = "2-bits-per-base",
    mapping: Dict[str, str] = None,
) -> Dict[str, any]:
    """
    Analyze how well Hamming distances are preserved in DNA encoding.
    
    For 2-bits-per-base with standard mapping, bit-level Hamming distance
    should translate directly to base-level Hamming distance.
    
    Args:
        codes: List of binary codes
        scheme: Encoding scheme
        mapping: Optional custom mapping
        
    Returns:
        Dictionary with analysis results
    """
    encoder = BitToDNAEncoder(scheme=scheme, mapping=mapping)
    dna_sequences = [encoder.encode(code) for code in codes]
    
    from .axis_code_generator import hamming_distance
    
    results = {
        "n_codes": len(codes),
        "scheme": scheme,
        "bits_per_base": encoder.bits_per_base,
        "preservation_ratio": [],
    }
    
    # Sample pairwise comparisons
    n_samples = min(1000, len(codes) * (len(codes) - 1) // 2)
    import random
    
    for _ in range(n_samples):
        i, j = random.sample(range(len(codes)), 2)
        
        bit_distance = hamming_distance(codes[i], codes[j])
        dna_distance = hamming_distance_dna(dna_sequences[i], dna_sequences[j])
        
        # For 2-bits-per-base, distances should scale as:
        # dna_distance = bit_distance / 2 (in expectation)
        expected_dna_distance = bit_distance / encoder.bits_per_base
        
        if expected_dna_distance > 0:
            ratio = dna_distance / expected_dna_distance
            results["preservation_ratio"].append(ratio)
    
    if results["preservation_ratio"]:
        import numpy as np
        results["mean_preservation_ratio"] = np.mean(results["preservation_ratio"])
        results["std_preservation_ratio"] = np.std(results["preservation_ratio"])
    
    return results
