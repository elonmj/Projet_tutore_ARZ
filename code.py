import numpy as np
import matplotlib.pyplot as plt

def setup_plot(ax, title, xlabel="Position (x)", ylabel="Valeur Moyenne (U)"):
    """Configure les éléments d'un subplot."""
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    if "WENO" not in title and "Discrétisation" not in title:
        ax.legend()

def generate_fvm_discretization_figure():
    """Génère le schéma illustrant la discrétisation FVM."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Cellules et interfaces
    centers = [2, 4, 6]
    interfaces = [1, 3, 5, 7]
    labels = ["$C_{j-1}$", "$C_j$", "$C_{j+1}$"]
    
    # Valeurs moyennes
    values = [0.4, 0.7, 0.6]
    
    for i, center in enumerate(centers):
        ax.plot([interfaces[i], interfaces[i+1]], [values[i], values[i]], 'r-', lw=3)
        ax.plot(center, values[i], 'ro', markersize=8, label="Moyenne cellulaire $\mathbf{U}_j^n$" if i==1 else "")
        ax.text(center, -0.1, labels[i], ha='center', fontsize=14)
        ax.axvline(center, color='gray', linestyle='--', ymax=(values[i]+0.1)/1.2)
        ax.text(center, -0.25, f'$x_j$', ha='center', fontsize=12)

    for i, interface in enumerate(interfaces):
        ax.axvline(interface, color='k', linestyle='-')
        label_x = "$x_{j-3/2}$" if i==0 else f"$x_{{j{i-2:+.1f}}}$".replace('.0','').replace('+-','-')
        ax.text(interface, -0.25, label_x, ha='center', fontsize=12)
    
    # Flux
    ax.arrow(3, 0.3, 0, 0.3, head_width=0.2, head_length=0.1, fc='b', ec='b', lw=2)
    ax.text(3.2, 0.45, '$\mathcal{F}_{j-1/2}$', color='b', fontsize=12)
    ax.arrow(5, 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='b', ec='b', lw=2)
    ax.text(5.2, 0.45, '$\mathcal{F}_{j+1/2}$', color='b', fontsize=12)

    ax.set_xlim(0, 8)
    ax.set_ylim(-0.3, 1.2)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    setup_plot(ax, "Schéma de la Discrétisation par Volumes Finis", ylabel="")
    if ax.get_legend_handles_labels()[0]:
      ax.legend()
    
    plt.tight_layout()
    plt.savefig("figure_fvm_discretisation.png", dpi=300)
    plt.close()
    print("Figure 'figure_fvm_discretisation.png' générée.")


def generate_reconstruction_figures():
    """Génère les figures pour la reconstruction 1er ordre et WENO."""
    # --- Figure 1: Reconstruction 1er ordre ---
    fig_ax1, ax1 = plt.subplots(figsize=(8, 6))
    x_cont = np.linspace(0, 10, 500)
    y_cont = np.sin(x_cont * 0.8) * 0.4 + 0.5
    
    x_disc = np.linspace(0.5, 9.5, 10)
    y_disc = np.sin(x_disc * 0.8) * 0.4 + 0.5
    
    ax1.plot(x_cont, y_cont, 'k-', lw=2.5, alpha=0.8, label="Solution Physique")
    ax1.step(x_disc, y_disc, 'r-', where='mid', lw=2, label="Reconstruction Ordre 1")
    ax1.plot(x_disc, y_disc, 'ro', markersize=6)
    setup_plot(ax1, "Reconstruction Constante par Morceaux (Ordre 1)")
    plt.tight_layout()
    plt.savefig("figure_reconstruction_1er_ordre.png", dpi=300)
    plt.close(fig_ax1)
    print("Figure 'figure_reconstruction_1er_ordre.png' générée.")


    # --- Figure 2: Reconstruction WENO ---
    fig_ax2, ax2 = plt.subplots(figsize=(8, 6))
    x_cont = np.linspace(0, 10, 500)
    y_cont = np.sin(x_cont * 0.8) * 0.4 + 0.5
    x_stencil = np.arange(2.5, 8.5)
    y_stencil = np.sin(x_stencil * 0.8) * 0.4 + 0.5
    
    # Mettre en évidence les pochoirs pour la cellule j (center=4.5)
    stencils = {
        'S1': {'indices': [2, 3, 4], 'color': 'orange'},
        'S2': {'indices': [3, 4, 5], 'color': 'purple'},
        'S3': {'indices': [4, 5, 6], 'color': 'cyan'},
    }
    
    for name, s in stencils.items():
        indices_range = slice(s['indices'][0]-2, s['indices'][2]-1)
        ax2.plot(x_stencil[indices_range], y_stencil[indices_range], 'o--', 
                 color=s['color'], alpha=0.8, markersize=8,
                 label=f"Pochoir {name}")

    # Cellule centrale
    ax2.axvspan(4, 5, color='green', alpha=0.15, label="Cellule $C_j$")
    ax2.plot(x_cont, y_cont, 'g-', lw=3, label="Reconstruction WENO")
    ax2.plot(x_stencil, y_stencil, 'ko', markersize=5)
    
    setup_plot(ax2, "Reconstruction WENO et ses Pochoirs")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("figure_weno_reconstruction.png", dpi=300)
    plt.close(fig_ax2)
    print("Figure 'figure_weno_reconstruction.png' générée.")


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-whitegrid')
    generate_fvm_discretization_figure()
    generate_reconstruction_figures()