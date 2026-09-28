# Date: 25-06-2026
# Creator: Julia Wiergowska

# This model calculates the change of concentration of TFA
# in the 1D convection–diffusion–reaction model for TFA removal
# in a bacterial cellulose biofilter with immobilised β-lactoglobulin.

# Loading packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

#Parameters for a bioreactor
A_cross     = 0.0227          # m², cross-sectional area (pilot scale)
diameter_biofilter = 0.1686
radius_biofilter = 0.084325
A_filter = np.pi * diameter_biofilter**2 / 4 #later change this for A_cross
H_biofilter = 2 #m, height / thickness of the cellulose; the value of H_biofilter is either changed to 0.1 or 2 for the whole model
V_total_biofilter = H_biofilter * A_cross #with the height of 3m - = 0.06699 m³ is around 67 L

#Hydrodynamics
FG = [0.1]  #liquid flow rates through the reactor (m³/h), range: 0.05 – 0.2
#Biofilter parameters
a = 650  #specific surface area of cellulose (m⁻¹), a value between 300-1000
ep = 0.95  #particle porosity
phi = 0.98  #sphericity
tp = 1.23 * ((1 - ep) ** (4 / 3) / (ep * phi))  #tortuosity
psize = 1.25e-3          # m (midpoint between 1000–1500 µm)
ncp = 3

#Mass transfer
D_TFA = 3.5e-6  #diffusion coefficient (m²/h)
D_TFA_effect = D_TFA * (ep / tp)  #effective intra-particle diffusivity (m²/h)
# D_TFA is reduced by porosity and tortuosity;

#External liquid mass transfer coefficient kL (m/h)
water_flow_rate = 0.1 #m³/h;
rho_water = 1000  #kg/m³
mu_water = 3.6  #kg/(m·h)
nu_water = mu_water / rho_water  # kinematic viscosity (m²/h) -> how fast water diffuses in the fluid
Sc = nu_water / D_TFA  # Schmidt number, defined as the ratio of momentum diffusivity (kinematic viscosity) to mass diffusivity
Re_p = (water_flow_rate * psize) / (a * V_total_biofilter * nu_water)  # particle Reynolds, ratio of inertial
# to viscous forces around the particle
#effective flow intensity around particles in a packed bed, from the Ranz-Marshall correlation
Sh = 2 + 0.6 * Re_p ** 0.5 * Sc ** (1 / 3)
#liquid film mass transfer coefficient (m/h)
kL = Sh * D_TFA / psize

#Material loading
c_xp = 10000 #binding-protein B-lactoglobulin density (g_DW/m³ particle)
L_diffusion = 5e-5 #m, thin hydrogel coating
# m; intraparticle diffusion length = bead radius (HSDM convention)
#                           # = 6.25e-4 m for psize = 1.25e-3 m
n = 1  #binding stoichiometry (mol TFA / mol β-lg); there is one primary binding site
weight_betalacto = 18300   # g/mol (molecular weight of β-lactoglobulin)
#maximum binding of the protein - q_max_specific
q_max_specific = n / weight_betalacto
# explanation of q_max: 1.0 / 18300 = 5.46e-5 mol TFA / g protein

#Binding kinetics (Langmuir)
#Kd of β-lactoglobulin binding to TFA; Kd = k_off/k_on [mol/L → converting to [mol/m³]
#k_on is the reverse rate constant and k_off is the forward rate constant.
#Kd = Km
Kd = 8.6e-3 * 1e3  # 8.6 mM +/- 2 mMmM -> mol/m³ +/- 2.0 value based on the [McLean, 2026] -> mol

#Q_max - maximum volumetric capacity of the layer; quantity that is most often used in mass transport equations for biofilters
Q_max = q_max_specific * c_xp #mol/m3, this comes from mol/g * g/m³

#initial TFA concentrations to go through (mol/m³)
#0.23 µg/L ≈ 0.23e-6 g/L ÷ 114 g/mol × 1000 L/m³ ≈ 2.02e-6 mol/m³ in the final  of the loop
MW_TFA = 114  # g/mol
#initial TFA concentrations (µg/L) -> convert it to Kd values consistency
C0_ugL = np.array([0.5, 2.2, 4.1])
#but 0.5 is our target -> we should model a system for 0.5 threshold and 0 for the last graph
#convert to mol/m³
CTFA0_list = (C0_ugL * 1e-6 / MW_TFA * 1e3) #-> mol/m3
#so CTFAO_list is the list of initial concentrations -> mol


#setting up empty arrays to save
data_Cg = []  #TFA concentration [mol/m³ vs V]
data_E = []  #removal efficiency (%)

# if Ji0 >> actual TFA load, binding proteins are in vast excess (validates QSS assumption)
#Ji0 = (Vmax * CTFA0_list.mean()) * c_xp / L  # initial flux (mol/m²·h), Eq. 1
#print(f"Initial inlet flux Ji0 = {Ji0:.3e} mol/(m²·h)")

# ODE model loop
for Fg in FG:
    for CTFA0 in CTFA0_list: # loop over initial TFA concentrations
        # combined mass transfer coefficient (external + intra-particle)


        def model(Cg, z): #Cg - current state vector, z - current position along the filter, z is not dependent of temp, tortuosity, etc.
            C_bulk = max(Cg[0], 0.0)
            C_p = max(Cg[1], 0.0)
            #1 / K_overall = 1 / k_L + L / D_eff
#the overall mass-transfer coefficient.
#analogous to resistors in series, Rtotal = Rfilm + Rdiffusion
            K_overall = 1 / (1 / kL + L_diffusion / D_TFA_effect)
            J_ext = K_overall * a * (C_bulk - C_p)
            # Langmuir isotherm
            r_bind = Q_max * C_p**ncp / (Kd**ncp + C_p**ncp)  # mol/m³ particle, an adsorbed amount

            dC_bulk_dz = -(A_filter / Fg) * J_ext
            #Change inside particle = incoming mass − adsorbed mass
            dC_p_dz = (A_filter / Fg) * (J_ext - (1 - ep) * r_bind)

            return [dC_bulk_dz, dC_p_dz]
        z_span = np.linspace(0, H_biofilter, 10000)
        Cg_sol = odeint(model, [CTFA0, 0.0], z_span)

        C_TFA = np.maximum(Cg_sol[:, 0], 0.0)
        data_Cg.append(C_TFA)

        E = (CTFA0 - C_TFA) / CTFA0 * 100
        data_E.append(E)

        target = 0.10 * CTFA0
        idx_90 = np.argmax(C_TFA <= target)
        if idx_90 > 0:
            L_90 = z_span[idx_90]
            print(f"C0 = {CTFA0 * MW_TFA * 1e6 / 1e3:.2f} µg/L → "
                  f"90% removal at z = {L_90:.4f} m")

# Kd sensitivity analysis
Kd_values     = np.logspace(-8, 2, 20)   # Kd = 10–50 nM/m3 generating 20 numbers
removal_final = []
CTFA0_sens    = CTFA0_list[1]
Fg_sens       = FG[0]

for Kd_test in Kd_values:
    K_overall_s = 1 / (1/kL + L_diffusion/D_TFA_effect)

    def model_sens(Cg, z):
        C_bulk = max(Cg[0], 0.0)
        C_p    = max(Cg[1], 0.0)

        J_ext  = K_overall_s * a * (C_bulk - C_p)
        r_bind = Q_max * C_p / (Kd_test + C_p)

        dC_bulk_dz = -(A_filter / Fg_sens) * J_ext
        dC_p_dz    =  (A_filter / Fg_sens) * (J_ext - (1 - ep) * r_bind)
        return [dC_bulk_dz, dC_p_dz]

    sol   = odeint(model_sens, [CTFA0_sens, 0.0], z_span)
    C_out = max(sol[-1, 0], 0.0)
    removal_final.append((CTFA0_sens - C_out) / CTFA0_sens * 100)

# ── c_xp sensitivity analysis ─────────────────────────────────────────────────
# (g_DW/m³ particle); nominal value is 10000
# varying over a range: 100 – 100000 g/m³
cxp_values      = np.logspace(2, 5, 20)   # 100 – 100,000 g_DW/m³ generating 20 numbers
removal_cxp     = []
Fg_sens         = FG[0]

for cxp_test in cxp_values:
    Q_max_test  = q_max_specific * cxp_test   # recompute Q_max for this c_xp
    K_overall_cxp = 1 / (1/kL + L_diffusion/D_TFA_effect)

    def model_cxp(Cg, z):
        C_bulk = max(Cg[0], 0.0)
        C_p    = max(Cg[1], 0.0)

        J_ext  = K_overall_cxp * a * (C_bulk - C_p)
        r_bind = Q_max_test * C_p / (Kd + C_p)   # nominal Kd, varying Q_max

#The factor (1-ep) converts from "per m3 particle" to "per m3 bulk volume"
        #because the fraction of the bed is 1 - ep.
        #(1-ep)*r_bind has units

        dC_bulk_dz = -(A_filter / Fg_sens) * J_ext
        dC_p_dz    =  (A_filter / Fg_sens) * (J_ext - (1 - ep) * r_bind)
        return [dC_bulk_dz, dC_p_dz]

    sol   = odeint(model_cxp, [CTFA0_sens, 0.0], z_span)
    C_out = max(sol[-1, 0], 0.0)
    removal_cxp.append((CTFA0_sens - C_out) / CTFA0_sens * 100)

# Plotting
# Plotting: 4 graphs in one figure
sns.set_theme(style="whitegrid", palette="muted")
colors = sns.color_palette("tab20", len(CTFA0_list))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# Graph 1: Concentration profile
for idx, (CTFA0, C0_label) in enumerate(zip(CTFA0_list, C0_ugL)):
    C_plot_ugL = data_Cg[idx] * MW_TFA * 1e6 / 1e3
    ax1.plot(z_span, C_plot_ugL,
             color=colors[idx],
             label=f"{C0_label} ug/L")

ax1.set_xlabel("Biofilter length (m)")
ax1.set_ylabel("TFA concentration (ug/L)")
ax1.set_title("TFA concentration along biofilter")
ax1.legend()
ax1.grid(True)

# Graph 2: Removal efficiency
for idx, C0_label in enumerate(C0_ugL):
    ax2.plot(z_span, data_E[idx],
             color=colors[idx],
             label=f"{C0_label} ug/L")

ax2.set_xlabel("Biofilter length (m)")
ax2.set_ylabel("Removal efficiency (%)")
ax2.set_title("TFA removal efficiency")
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True)

# Graph 3: Kd sensitivity
ax3.semilogx(Kd_values, removal_final, marker="o")
ax3.axvline(Kd, color="red", linestyle="--",
            label=f"Nominal Kd = {Kd:.1f}")
ax3.set_xlabel("Kd (mol/m3)")
ax3.set_ylabel("Final removal efficiency (%)")
ax3.set_title("Sensitivity to Kd")
ax3.set_ylim(0, 105)
ax3.legend()
ax3.grid(True)

# Graph 4: Protein loading sensitivity
ax4.semilogx(cxp_values, removal_cxp, marker="o")
ax4.axvline(c_xp, color="red", linestyle="--",
            label=f"Nominal c = {c_xp:.0f}")
ax4.set_xlabel("Binding-protein density (g/m3 particle)")
ax4.set_ylabel("Final removal efficiency (%)")
ax4.set_title("Sensitivity to protein loading")
ax4.set_ylim(0, 105)
ax4.legend()
ax4.grid(True)

fig.tight_layout(pad=3.0)

fig.savefig(
    r"C:\Users\julia\OneDrive - Wageningen University & Research"
    r"\Documenten\igem\computational_part\biofilter_model.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

