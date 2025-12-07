def build_regional_vulnerability_plot(rnfl_shap: np.ndarray, gcipl_shap: np.ndarray):
    """
    Polar maps for GCIPL sectors (left) and RNFL clock-hours (right).

    - Orientation for OD:
        * Temporal retina is on the LEFT, nasal on the RIGHT.
        * 12H is superior, 6H is inferior (bottom).
    - Color:
        * Red: more resilient / protective (contributes to younger optic nerve age)
        * Blue: more vulnerable / harmful (contributes to older optic nerve age)
    """
    # Flip sign so that protective (age-decreasing) regions appear red
    rnfl_vals = -rnfl_shap
    gc_vals = -gcipl_shap

    vmax = max(np.max(np.abs(rnfl_vals)), np.max(np.abs(gc_vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # high=red, low=blue

    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor("#111827")

    # ---------- Left: GCIPL sectors ----------
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax1.set_facecolor("#111827")
    # 0° = 위(Sup), 시계방향 증가 → 90° = 오른쪽(Nasal), 180° = 아래(Inf), 270° = 왼쪽(Temporal)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)

    num_gc = 6
    width_gc = 2 * pi / num_gc

    # GCIPL_FEATURES 순서:
    # [TempSup, Sup, NasSup, NasInf, Inf, TempInf]
    # 각 sector의 중심 각도(도 단위)를 지정해서 Inf가 180°(6시 방향)에 오도록 설정
    gc_sectors = [
        ("NasInf",  gc_vals[3], 120.0),  # infero-nasal
        ("NasSup",  gc_vals[2],  60.0),  # supero-nasal
        ("Sup",     gc_vals[1],   0.0),  # superior (12시)
        ("TempSup", gc_vals[0], 300.0),  # supero-temporal
        ("TempInf", gc_vals[5], 240.0),  # infero-temporal
        ("Inf",     gc_vals[4], 180.0),  # inferior (6시, 맨 아래)
    ]

    for label, val, center_deg in gc_sectors:
        start_rad = math.radians(center_deg) - width_gc / 2.0
        ax1.bar(
            start_rad,
            1.0,
            width=width_gc,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    gc_tick_angles = [math.radians(c_deg) for _, _, c_deg in gc_sectors]
    gc_labels = [label for label, _, _ in gc_sectors]
    ax1.set_xticks(gc_tick_angles)
    ax1.set_xticklabels(gc_labels, color="white", fontsize=9)
    ax1.set_yticklabels([])
    ax1.set_title("GCIPL sector\ncontribution (OD)", color="white", fontsize=12, pad=10)

    # ---------- Middle: color bar ----------
    cax = fig.add_subplot(1, 3, 2)
    cax.set_visible(False)
    cb_ax = fig.add_axes([0.44, 0.18, 0.03, 0.64])  # [left, bottom, width, height]
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cb_ax,
    )
    cb_ax.set_ylabel("SHAP value\n(impact on optic nerve age)", color="white", fontsize=9)
    cb_ax.yaxis.set_tick_params(color="white")
    plt.setp(cb_ax.get_yticklabels(), color="white", fontsize=8)
    cb_ax.text(
        0.5,
        1.05,
        "More resilient (healthier / thicker)",
        color="white",
        fontsize=8,
        ha="center",
        transform=cb_ax.transAxes,
    )
    cb_ax.text(
        0.5,
        -0.12,
        "More vulnerable (older / thinner)",
        color="white",
        fontsize=8,
        ha="center",
        transform=cb_ax.transAxes,
    )

    # ---------- Right: RNFL clock-hour ----------
    ax2 = fig.add_subplot(1, 3, 3, polar=True)
    ax2.set_facecolor("#111827")
    # 동일하게 0°=12H, 시계방향으로 3H(90°), 6H(180°), 9H(270°)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    num = 12
    width = 2 * pi / num

    hours = list(range(1, 13))  # 1H~12H
    for h in hours:
        val = rnfl_vals[h - 1]          # RNFL_FEATURES: CLOCKHOUR_1~12
        center_deg = (h % 12) * 30.0    # 12H→0°, 1H→30°, ..., 6H→180°(아래)
        start_rad = math.radians(center_deg) - width / 2.0
        ax2.bar(
            start_rad,
            1.0,
            width=width,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    hour_centers = [math.radians((h % 12) * 30.0) for h in hours]
    hour_labels = [f"{h}H" for h in hours]
    ax2.set_xticks(hour_centers)
    ax2.set_xticklabels(hour_labels, color="white", fontsize=9)
    ax2.set_yticklabels([])
    ax2.set_title("RNFL clock-hour\ncontribution (OD)", color="white", fontsize=12, pad=10)

    plt.tight_layout()
    return fig

