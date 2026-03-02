"""
Generate LinkedIn-Ready Comparison Plots
=========================================
Reads results.json and creates 4 publication-quality figures.

Usage:
    python plot_results.py                    # Default
    python plot_results.py --input results.json --output plots/
"""

import json
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ======================================================================
# Styling — Dark, modern, LinkedIn-ready
# ======================================================================

COLORS = {
    'mlp': '#4ECDC4',       # Teal
    'kan': '#FF6B6B',       # Coral
    'mlp_light': '#4ECDC480',
    'kan_light': '#FF6B6B80',
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'text': '#e8e8e8',
    'grid': '#ffffff15',
    'accent': '#ffd93d',
}

def setup_style():
    """Apply dark premium style."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': '#ffffff30',
        'axes.labelcolor': COLORS['text'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'font.size': 12,
        'font.family': 'sans-serif',
        'legend.facecolor': '#ffffff10',
        'legend.edgecolor': '#ffffff20',
        'legend.fontsize': 11,
    })


# ======================================================================
# Individual Plots
# ======================================================================

def plot_training_loss(results, output_dir):
    """Plot 1: Training loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, color in [('mlp', COLORS['mlp']), ('kan', COLORS['kan'])]:
        if name not in results['experiments']:
            continue
        exp = results['experiments'][name]
        epochs = [e['epoch'] for e in exp['epochs']]
        train_loss = [e['train']['loss'] for e in exp['epochs']]
        val_loss = [e['val']['loss'] for e in exp['epochs']]
        
        ax.plot(epochs, train_loss, color=color, linewidth=2.5, 
                label=f'{name.upper()} Train', marker='o', markersize=6)
        ax.plot(epochs, val_loss, color=color, linewidth=2.5, linestyle='--',
                label=f'{name.upper()} Val', marker='s', markersize=5, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=18, fontweight='bold',
                 pad=20, color=COLORS['accent'])
    ax.legend(loc='upper right', framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def plot_perplexity(results, output_dir):
    """Plot 2: Perplexity comparison over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, color in [('mlp', COLORS['mlp']), ('kan', COLORS['kan'])]:
        if name not in results['experiments']:
            continue
        exp = results['experiments'][name]
        epochs = [e['epoch'] for e in exp['epochs']]
        val_ppl = [e['val']['ppl'] for e in exp['epochs']]
        
        ax.plot(epochs, val_ppl, color=color, linewidth=3, 
                label=f'{name.upper()}', marker='o', markersize=8)
        
        # Annotate final value
        ax.annotate(f'{val_ppl[-1]:.1f}', xy=(epochs[-1], val_ppl[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    color=color, fontsize=13, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity (↓ better)', fontsize=14, fontweight='bold')
    ax.set_title('Validation Perplexity', fontsize=18, fontweight='bold',
                 pad=20, color=COLORS['accent'])
    ax.legend(loc='upper right', fontsize=14, framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'perplexity.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def plot_params_and_speed(results, output_dir):
    """Plot 3: Parameter count and throughput comparison bars."""
    if len(results['experiments']) < 2:
        print("  ⚠ Skipping param/speed plot (need both models)")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = []
    params = []
    speeds = []
    times = []
    colors = []
    
    for name in ['mlp', 'kan']:
        if name in results['experiments']:
            exp = results['experiments'][name]
            models.append(name.upper())
            params.append(exp['params'])
            # Average throughput across all epochs
            avg_speed = np.mean([e['train']['tokens_per_sec'] for e in exp['epochs']])
            speeds.append(avg_speed)
            times.append(exp['total_time_seconds'])
            colors.append(COLORS[name])
    
    # Bar 1: Parameters
    bars1 = ax1.bar(models, [p / 1000 for p in params], color=colors, 
                     width=0.5, edgecolor='white', linewidth=0.5)
    for bar, p in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{p:,}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=COLORS['text'])
    ax1.set_ylabel('Parameters (K)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Size', fontsize=15, fontweight='bold', color=COLORS['accent'])
    
    # Bar 2: Throughput
    bars2 = ax2.bar(models, speeds, color=colors,
                     width=0.5, edgecolor='white', linewidth=0.5)
    for bar, s in zip(bars2, speeds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speeds)*0.02,
                f'{s:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=COLORS['text'])
    ax2.set_ylabel('Tokens / Second', fontsize=13, fontweight='bold')
    ax2.set_title('Training Throughput', fontsize=15, fontweight='bold', color=COLORS['accent'])
    
    # Style
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Efficiency Comparison', fontsize=18, fontweight='bold',
                 y=1.02, color=COLORS['accent'])
    plt.tight_layout()
    path = os.path.join(output_dir, 'efficiency.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


def plot_summary_panel(results, output_dir):
    """Plot 4: Combined 2x2 summary panel — perfect for LinkedIn."""
    if len(results['experiments']) < 2:
        print("  ⚠ Skipping summary (need both models)")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KAN vs MLP as Transformer FFN — Comparison',
                 fontsize=22, fontweight='bold', y=0.98, color=COLORS['accent'])
    
    mlp_exp = results['experiments']['mlp']
    kan_exp = results['experiments']['kan']
    
    # --- Panel 1: Loss Curves ---
    ax = axes[0, 0]
    for name, exp, color in [('MLP', mlp_exp, COLORS['mlp']), ('KAN', kan_exp, COLORS['kan'])]:
        epochs = [e['epoch'] for e in exp['epochs']]
        ax.plot(epochs, [e['train']['loss'] for e in exp['epochs']],
                color=color, linewidth=2.5, label=f'{name} Train', marker='o', markersize=4)
        ax.plot(epochs, [e['val']['loss'] for e in exp['epochs']],
                color=color, linewidth=2, linestyle='--', label=f'{name} Val', alpha=0.7)
    ax.set_title('Loss', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # --- Panel 2: Perplexity ---
    ax = axes[0, 1]
    for name, exp, color in [('MLP', mlp_exp, COLORS['mlp']), ('KAN', kan_exp, COLORS['kan'])]:
        epochs = [e['epoch'] for e in exp['epochs']]
        val_ppl = [e['val']['ppl'] for e in exp['epochs']]
        ax.plot(epochs, val_ppl, color=color, linewidth=3, label=name, marker='o', markersize=6)
        ax.annotate(f'{val_ppl[-1]:.1f}', xy=(epochs[-1], val_ppl[-1]),
                    xytext=(8, 5), textcoords='offset points',
                    color=color, fontsize=11, fontweight='bold')
    ax.set_title('Validation Perplexity (↓ better)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # --- Panel 3: Accuracy ---
    ax = axes[1, 0]
    for name, exp, color in [('MLP', mlp_exp, COLORS['mlp']), ('KAN', kan_exp, COLORS['kan'])]:
        epochs = [e['epoch'] for e in exp['epochs']]
        acc = [e['val']['accuracy'] for e in exp['epochs']]
        ax.plot(epochs, acc, color=color, linewidth=3, label=name, marker='s', markersize=6)
        ax.annotate(f'{acc[-1]:.1f}%', xy=(epochs[-1], acc[-1]),
                    xytext=(8, 5), textcoords='offset points',
                    color=color, fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy (↑ better)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy %')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # --- Panel 4: Summary Table ---
    ax = axes[1, 1]
    ax.axis('off')
    
    mlp_final = mlp_exp['epochs'][-1]
    kan_final = kan_exp['epochs'][-1]
    
    table_data = [
        ['Metric', 'MLP', 'KAN'],
        ['Parameters', f"{mlp_exp['params']:,}", f"{kan_exp['params']:,}"],
        ['Final Val PPL', f"{mlp_final['val']['ppl']:.1f}", f"{kan_final['val']['ppl']:.1f}"],
        ['Final Val Acc', f"{mlp_final['val']['accuracy']:.1f}%", f"{kan_final['val']['accuracy']:.1f}%"],
        ['Throughput', f"{mlp_final['train']['tokens_per_sec']:,.0f}/s", 
         f"{kan_final['train']['tokens_per_sec']:,.0f}/s"],
        ['Total Time', f"{mlp_exp['total_time_seconds']:.0f}s", 
         f"{kan_exp['total_time_seconds']:.0f}s"],
    ]
    
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.3, 0.3],
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#ffffff30')
        if row == 0:
            cell.set_facecolor(COLORS['accent'])
            cell.set_text_props(color='#1a1a2e', fontweight='bold', fontsize=13)
        else:
            cell.set_facecolor(COLORS['panel'])
            cell.set_text_props(color=COLORS['text'])
            if col == 1:
                cell.set_text_props(color=COLORS['mlp'], fontweight='bold')
            elif col == 2:
                cell.set_text_props(color=COLORS['kan'], fontweight='bold')
    
    ax.set_title('Summary', fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'summary_panel.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots')
    parser.add_argument('--input', type=str, default='results.json')
    parser.add_argument('--output', type=str, default='plots')
    args = parser.parse_args()
    
    setup_style()
    
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_dir = os.path.join(script_dir, args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Reading {input_path}...")
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    print(f"🎨 Generating plots to {output_dir}/\n")
    
    plot_training_loss(results, output_dir)
    plot_perplexity(results, output_dir)
    plot_params_and_speed(results, output_dir)
    plot_summary_panel(results, output_dir)
    
    print(f"\n✅ All plots saved to {output_dir}/")
    print("📱 The summary_panel.png is ideal for LinkedIn!")


if __name__ == '__main__':
    main()
