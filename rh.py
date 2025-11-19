#!/usr/bin/env python3
"""
seqlogo_tool.py

Develop a simple sequence-logo tool using Position Weight Matrices (PWMs).
Supports:
 - Input: aligned FASTA per species OR PWM CSV files (rows=A,C,G,T; cols=positions)
 - Outputs: matplotlib sequence logo plots (one per species) and optional PNG files.

Dependencies: numpy, pandas, matplotlib
Usage examples:
  python seqlogo_tool.py --human human_aligned.fasta --rat rat_aligned.fasta --out_prefix human_vs_rat
  python seqlogo_tool.py --human_pwm human_pwm.csv --rat_pwm rat_pwm.csv --out_prefix pwm_compare

Author: ChatGPT (GPT-5 Thinking mini)
"""

import sys
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

ALPHABET = ['A', 'C', 'G', 'T']

def parse_fasta(path):
    """Simple FASTA parser returning list of sequences (uppercased, stripped)."""
    seqs = []
    with open(path, 'r') as fh:
        name = None
        buf = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    seqs.append(''.join(buf).upper())
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            seqs.append(''.join(buf).upper())
    return seqs

def validate_alignment(seqs):
    """Check all sequences have the same length."""
    if not seqs:
        raise ValueError("No sequences provided")
    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("Sequences are not aligned: differing lengths found")
    return L

def compute_count_matrix(seqs, pseudocount=1):
    """Compute counts of A,C,G,T at each position with pseudocount added."""
    L = validate_alignment(seqs)
    counts = np.zeros((4, L), dtype=float)  # rows in order A,C,G,T
    base_to_idx = {b:i for i,b in enumerate(ALPHABET)}
    for s in seqs:
        for i, ch in enumerate(s):
            if ch in base_to_idx:
                counts[base_to_idx[ch], i] += 1
            else:
                # treat unknown (N, -, others) as not counted
                pass
    # add pseudocount
    counts += pseudocount
    return counts

def counts_to_freqs(counts):
    """Convert counts matrix to frequency matrix per column."""
    col_sums = counts.sum(axis=0, keepdims=True)
    freqs = counts / col_sums
    return freqs

def read_pwm_csv(path):
    """
    Expect CSV with rows labeled A,C,G,T and columns 1..L (positions).
    Example:
    ,1,2,3
    A,0.2,0.1,0.6
    C,0.3,0.4,0.1
    G,0.3,0.2,0.2
    T,0.2,0.3,0.1
    """
    df = pd.read_csv(path, index_col=0)
    # enforce order A,C,G,T
    try:
        df = df.loc[ALPHABET]
    except KeyError:
        # If different ordering or extra rows, try to map
        missing = [b for b in ALPHABET if b not in df.index]
        if missing:
            raise ValueError(f"PWM CSV missing rows: {missing}")
        df = df.reindex(ALPHABET)
    arr = df.values.astype(float)
    # ensure columns sum to 1 (normalize)
    arr = arr / (arr.sum(axis=0, keepdims=True))
    return arr

def pwm_from_seqs(seqs, pseudocount=1):
    counts = compute_count_matrix(seqs, pseudocount=pseudocount)
    freqs = counts_to_freqs(counts)
    return freqs  # this is frequency matrix

def background_default():
    """Return default background freq (uniform)."""
    return np.array([0.25, 0.25, 0.25, 0.25])

def freqs_to_pwm_logodds(freqs, background=None, pseudocount=1e-9):
    """
    Compute log-odds PWM = log2(freq / background).
    freqs: 4 x L frequency matrix
    background: length-4 vector
    """
    if background is None:
        background = background_default()
    # avoid division by zero by adding tiny pseudocount
    pwm = np.log2((freqs + pseudocount) / background[:,None])
    return pwm

def compute_information_matrix(freqs, background=None):
    """
    Compute information content (bits) for each base at each position
    using formula per Schneider & Stephens:
    R_seq = log2(4) - (H_obs + small_sample_correction)
    where H_obs = -sum_b f_b * log2(f_b)
    and height of base b at position i = f_b * R_seq
    We'll implement small-sample correction (e.g., e_n) optionally using sequence count.
    """
    if background is None:
        background = background_default()
    # number of sequences approximated by column sum of counts (can't derive from freqs alone).
    # If user computed freqs from counts with pseudocounts, we can't reliably get N. We'll skip small sample correction.
    L = freqs.shape[1]
    H = -np.sum(np.where(freqs>0, freqs * np.log2(freqs), 0.0), axis=0)  # length L
    R_seq = np.log2(4) - H  # max 2 bits
    R_seq = np.maximum(R_seq, 0.0)
    # heights per base:
    heights = freqs * R_seq[np.newaxis, :]
    return heights, R_seq

def plot_sequence_logo(heights, ax=None, title=None, letters=ALPHABET, figsize=(10,3), savepath=None):
    """
    Draw a basic sequence logo from heights array (4 x L).
    Stacks positive heights for each column; letter order arranged so largest on top.
    This is a simple implementation that draws each letter as text scaled according to height.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    num_bases, L = heights.shape
    x = np.arange(1, L+1)
    max_height = np.max(heights.sum(axis=0)) if L>0 else 2.0
    y_bottom = np.zeros(L)

    # For each column, sort letters by height so draw bottom->top
    for i in range(L):
        # pair (base, height)
        pairs = [(letters[b], heights[b, i]) for b in range(num_bases)]
        # sort by height ascending so smallest drawn first (bottom) — but we want larger at top
        pairs_sorted = sorted(pairs, key=lambda p: p[1])
        y = 0.0
        for base, h in pairs_sorted:
            if h <= 1e-8:
                continue
            # Determine font size and text transform
            # We'll map height (bits) to relative font size. This mapping is heuristic.
            # max height (2 bits) -> font size 100 (large). Scale linearly.
            fontsize = 8 + (h / 2.0) * 92  # between 8 and ~100
            # Place text at (x[i], y + h/2) to center it in the current stack slice.
            ax.text(x[i], y + h/2.0, base,
                    fontsize=fontsize,
                    fontweight='bold',
                    ha='center', va='center',
                    family='DejaVu Sans')
            y += h

    ax.set_xlim(0.5, L+0.5)
    ax.set_ylim(0, max(2.0, np.max(heights.sum(axis=0))*1.05))
    ax.set_xlabel('Position')
    ax.set_ylabel('Bits')
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_yticks(np.linspace(0,2,5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')

    return ax

def plot_two_logos(heights1, heights2, labels=('Species1','Species2'), out_prefix=None):
    L1 = heights1.shape[1]
    L2 = heights2.shape[1]
    if L1 != L2:
        print("Warning: positions differ between inputs (L1=%d, L2=%d). Plotting separately." % (L1, L2))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, max(L1,L2)*0.5), 6))
    plot_sequence_logo(heights1, ax=axes[0], title=labels[0])
    plot_sequence_logo(heights2, ax=axes[1], title=labels[1])
    plt.tight_layout()
    if out_prefix:
        path = out_prefix + "_logos.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print("Saved logo figure to:", path)
    plt.show()

def build_from_fasta_paths(human_fasta, rat_fasta, pseudocount=1.0, background=None):
    human_seqs = parse_fasta(human_fasta)
    rat_seqs = parse_fasta(rat_fasta)
    # validations
    Lh = validate_alignment(human_seqs)
    Lr = validate_alignment(rat_seqs)
    if Lh != Lr:
        print("Warning: alignment lengths differ (human=%d, rat=%d). Logos will be computed separately." % (Lh, Lr))
    human_freqs = pwm_from_seqs(human_seqs, pseudocount=pseudocount)
    rat_freqs = pwm_from_seqs(rat_seqs, pseudocount=pseudocount)
    if background is None:
        bg = background_default()
    else:
        bg = np.asarray(background)
    # compute heights
    h_heights, h_R = compute_information_matrix(human_freqs, background=bg)
    r_heights, r_R = compute_information_matrix(rat_freqs, background=bg)
    return h_heights, r_heights

def build_from_pwm_paths(human_pwm_csv, rat_pwm_csv, background=None):
    human_freqs = read_pwm_csv(human_pwm_csv)
    rat_freqs = read_pwm_csv(rat_pwm_csv)
    if background is None:
        bg = background_default()
    else:
        bg = np.asarray(background)
    h_heights, h_R = compute_information_matrix(human_freqs, background=bg)
    r_heights, r_R = compute_information_matrix(rat_freqs, background=bg)
    return h_heights, r_heights

def main():
    parser = argparse.ArgumentParser(description="Sequence logo tool using PWMs for two species (human, rat).")
    # input options - either FASTA aligned sequences or PWM CSVs
    parser.add_argument('--human', help='Aligned FASTA for human (one sequence per line allowed).')
    parser.add_argument('--rat', help='Aligned FASTA for rat.')
    parser.add_argument('--human_pwm', help='Human PWM CSV (rows A,C,G,T).')
    parser.add_argument('--rat_pwm', help='Rat PWM CSV (rows A,C,G,T).')
    parser.add_argument('--pseudocount', type=float, default=1.0, help='Pseudocount to add to counts (when using FASTA). Default=1.0')
    parser.add_argument('--out_prefix', default='seqlogo', help='Prefix for output files')
    parser.add_argument('--no_show', action='store_true', help='Do not show interactive plots (save only).')
    args = parser.parse_args()

    # Determine data source
    if args.human and args.rat:
        h_heights, r_heights = build_from_fasta_paths(args.human, args.rat, pseudocount=args.pseudocount)
    elif args.human_pwm and args.rat_pwm:
        h_heights, r_heights = build_from_pwm_paths(args.human_pwm, args.rat_pwm)
    else:
        print("ERROR: Provide either --human and --rat FASTA files OR --human_pwm and --rat_pwm CSV files.")
        parser.print_help()
        sys.exit(1)

    # Plot and save combined figure
    out_prefix = args.out_prefix
    fig, axes = plt.subplots(2, 1, figsize=(max(8, h_heights.shape[1]*0.5), 6))
    plot_sequence_logo(h_heights, ax=axes[0], title='Human', savepath=None)
    plot_sequence_logo(r_heights, ax=axes[1], title='Rat', savepath=None)
    plt.tight_layout()
    outpath = out_prefix + "_human_vs_rat.png"
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print("Saved combined logo:", outpath)
    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    main()
