import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings('ignore')


def electric_field(x, y, charges):
    """计算电场强度"""
    k = 8.99e9
    Ex = np.zeros_like(x)
    Ey = np.zeros_like(x)

    for x0, y0, q in charges:
        rx = x - x0
        ry = y - y0
        r = np.sqrt(rx ** 2 + ry ** 2)
        r = np.maximum(r, 0.08)

        E_mag = k * np.abs(q) / r ** 2
        direction = np.sign(q)

        Ex += direction * E_mag * rx / r
        Ey += direction * E_mag * ry / r

    return Ex, Ey


def calculate_potential(x, y, charges):
    """计算电势"""
    k = 8.99e9
    V = np.zeros_like(x)
    for x0, y0, q in charges:
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        r = np.maximum(r, 0.08)
        V += k * q / r
    return V


def draw_charge(ax, x, y, q):
    """绘制电荷符号"""
    if q > 0:
        circle = Circle((x, y), 0.1, facecolor='#FF6347', edgecolor='#DC143C',
                        linewidth=2, alpha=0.9, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, '+', fontsize=18, fontweight='bold',
                ha='center', va='center', color='white', zorder=11)
    else:
        circle = Circle((x, y), 0.1, facecolor='#4169E1', edgecolor='#00008B',
                        linewidth=2, alpha=0.9, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, '-', fontsize=18, fontweight='bold',
                ha='center', va='center', color='white', zorder=11)


class InteractiveElectricField:
    def __init__(self):
        self.charges = []
        self.q0 = 1e-9
        self.xlim = (-2.5, 2.5)
        self.ylim = (-2, 2)

        self.vmin_fixed = -50
        self.vmax_fixed = 50

        self.cbar1 = None
        self.cbar2 = None
        self.cax1 = None
        self.cax2 = None

        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.setup_plots()
        self.update_plot()

        # 调整布局，为标题预留空间
        self.fig.subplots_adjust(top=0.92)
        plt.show()

    def setup_plots(self):
        self.ax1.set_xlim(self.xlim)
        self.ax1.set_ylim(self.ylim)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlabel('x (m)', fontsize=11)
        self.ax1.set_ylabel('y (m)', fontsize=11)
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.ax1.set_title('Electric Field Lines', fontsize=13)

        self.ax2.set_xlim(self.xlim)
        self.ax2.set_ylim(self.ylim)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('x (m)', fontsize=11)
        self.ax2.set_ylabel('y (m)', fontsize=11)
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.ax2.set_title('Potential Distribution', fontsize=13)

        instructions = (
            "Instructions:\n"
            "Left click: Add + charge\n"
            "Right click: Add - charge\n"
            "C: Clear all\n"
            "D: Dipole example\n"
            "T: Triple example"
        )
        self.ax1.text(0.02, 0.98, instructions, transform=self.ax1.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.setup_plots()

        if not self.charges:
            self.ax1.text(0, 0, 'Click to add charges\nLeft: +   Right: -',
                          ha='center', va='center', fontsize=14, color='gray',
                          transform=self.ax1.transAxes)
            self.ax2.text(0, 0, 'Click to add charges\nLeft: +   Right: -',
                          ha='center', va='center', fontsize=14, color='gray',
                          transform=self.ax2.transAxes)
            self.fig.suptitle('Interactive Electric Field Simulation | Positive: 0 | Negative: 0',
                              fontsize=14, fontweight='bold')
            self.fig.canvas.draw_idle()
            return

        n_pos = sum(1 for _, _, q in self.charges if q > 0)
        n_neg = sum(1 for _, _, q in self.charges if q < 0)

        # 创建网格
        x = np.linspace(self.xlim[0], self.xlim[1], 40)
        y = np.linspace(self.ylim[0], self.ylim[1], 40)
        X, Y = np.meshgrid(x, y)

        # 计算电场
        Ex, Ey = electric_field(X, Y, self.charges)

        # 计算电场强度（用于颜色）
        E_mag = np.sqrt(Ex ** 2 + Ey ** 2)

        # 归一化方向（用于流线图）
        E_norm = np.maximum(E_mag, 1e-10)
        Ex_norm = Ex / E_norm
        Ey_norm = Ey / E_norm

        # 使用 matplotlib 内置的 streamplot
        stream = self.ax1.streamplot(X, Y, Ex_norm, Ey_norm,
                                     color=E_mag, cmap='plasma',
                                     linewidth=1.0, density=1.5,
                                     arrowsize=1.0, arrowstyle='->',
                                     minlength=0.1)

        # 左边图颜色条
        if self.cbar1 is not None:
            try:
                self.cbar1.remove()
            except:
                pass

        divider1 = make_axes_locatable(self.ax1)
        self.cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        self.cbar1 = self.fig.colorbar(stream.lines, cax=self.cax1, label='E-field (N/C)')

        # 电势计算
        x_fine = np.linspace(self.xlim[0], self.xlim[1], 100)
        y_fine = np.linspace(self.ylim[0], self.ylim[1], 100)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        V = calculate_potential(X_fine, Y_fine, self.charges)

        # 电势云图
        im = self.ax2.contourf(X_fine, Y_fine, V, levels=20, cmap='RdBu_r', alpha=0.6,
                               vmin=self.vmin_fixed, vmax=self.vmax_fixed)

        # 等势线
        contour = self.ax2.contour(X_fine, Y_fine, V, levels=8, colors='black',
                                   linewidths=0.8, linestyles='--', alpha=0.4)
        self.ax2.clabel(contour, inline=True, fontsize=8, fmt='%.0f')

        # 绘制电荷
        for x0, y0, q in self.charges:
            draw_charge(self.ax1, x0, y0, q)
            draw_charge(self.ax2, x0, y0, q)

        # 标题（放在最上面）
        self.fig.suptitle(f'Interactive Electric Field Simulation | Positive: {n_pos} | Negative: {n_neg}',
                          fontsize=14, fontweight='bold')

        # 右边图颜色条
        if self.cbar2 is not None:
            try:
                self.cbar2.remove()
            except:
                pass

        divider2 = make_axes_locatable(self.ax2)
        self.cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        self.cbar2 = self.fig.colorbar(im, cax=self.cax2, label='Potential (V)')

        # 调整布局：为标题预留空间，不压缩图的尺寸
        self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.94)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in [self.ax1, self.ax2]:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.charges.append([x, y, self.q0])
            print(f"Added + charge at ({x:.2f}, {y:.2f})")
        elif event.button == 3:
            self.charges.append([x, y, -self.q0])
            print(f"Added - charge at ({x:.2f}, {y:.2f})")
        self.update_plot()

    def on_key(self, event):
        if event.key == 'c':
            self.charges.clear()
            print("Cleared all charges")
            self.update_plot()
        elif event.key == 'd':
            self.charges.clear()
            self.charges.append([0.8, 0, self.q0])
            self.charges.append([-0.8, 0, -self.q0])
            print("Added dipole example (+q at 0.8, -q at -0.8)")
            self.update_plot()
        elif event.key == 't':
            self.charges.clear()
            self.charges.append([-0.8, 0, self.q0])
            self.charges.append([0.8, 0, self.q0])
            self.charges.append([0, 0.8, -self.q0])
            print("Added triple charge example")
            self.update_plot()


def main():
    print("=" * 60)
    print("Interactive Electric Field Simulation")
    print("=" * 60)
    print("Instructions:")
    print("  • Left click -> Add positive charge (+1 nC)")
    print("  • Right click -> Add negative charge (-1 nC)")
    print("  • Press 'C' -> Clear all charges")
    print("  • Press 'D' -> Add dipole example")
    print("  • Press 'T' -> Add triple charge example")
    print("=" * 60)
    print("")
    print("Using matplotlib's built-in streamplot (no duplicate lines!)")
    print("=" * 60)

    app = InteractiveElectricField()


if __name__ == "__main__":
    main()