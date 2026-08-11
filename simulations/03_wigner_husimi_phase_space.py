
# This script constructs a tissue-configurable, quasi-classical, physics-based simulator of voxel-scale nuclear magnetic resonance Bloch dynamics. (Ozymandias McEvoy)
# The RF excitation is modeled as an ideal, instantaneous, single-flip-angle unitary rotation.
# Longitudinal and transverse relaxation are modeled through idealized T1 and T2 relaxation channels.
# The script simulates voxel-level magnetization dynamics over several multiples of the selected tissue's T2 and visualizes the resulting transverse decay.


# Import Relevent Packages
import scipy.constants as const
from scipy.linalg import expm
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm



# ------------------------------------------------------------ Experimental Knobs and Biological Constants------------------------------------------------------------ #

# Relaxation Times, TRs, and TEs were estimated from the literature to be physically reasonable to allow for realistic simulation. This section can be easily modified #
# to accurately reflect specific experimental conditions 																											   #



# Physical Proton density [protons / m^3]
nH2O_ref      = 6.666e28;   	   # pure-water reference at ~25 C

# Relevent Field Strengths (Select Based on Design Choice)
#B0Mag = 3                          # Tesla
B0Mag  = 7                          # Tesla
#B0Mag = 9.4                          # Tesla 

# Tissue of Interest (Select Based off design choice)
tissue = 'Gray_Matter'
#tissue = 'White_Matter'
#tissue = 'Arterial_Blood'
#tissue = 'Cer_Spin_Fluid'


# Relative Proton Density Determined by tissue choice
if tissue == 'Gray_Matter':
    # Relative proton density / water-content-like values
	# normalized so CSF (approximately pure water) = 1.00
      PD = 0.84
      # Scaled to true number
      nH = PD * nH2O_ref
      
if tissue == 'White_Matter':
    # Relative proton density / water-content-like values
	# normalized so CSF (approximately pure water) = 1.00
      PD = 0.69
      # Scaled to true number
      nH = PD * nH2O_ref
      
if tissue == 'Arterial_Blood':
    # Relative proton density / water-content-like values
	# normalized so CSF (approximately pure water) = 1.00
      PD = 0.83
      # Scaled to true number
      nH = PD * nH2O_ref
      
if tissue == 'Cer_Spin_Fluid':
    # Relative proton density / water-content-like values
	# normalized so CSF (approximately pure water) = 1.00
      PD = 1.00
      # Scaled to true number
      nH = PD * nH2O_ref


# Main Magnetic Field Declaration
Bx = 0
By = 0
Bz = B0Mag
# Concatenate Field
B0 = np.array([Bx,By,Bz])
B0_norm = np.sqrt(Bx**2 + By**2 + Bz**2)


if B0Mag == 3:
	# Reasonable 3T TR's (Select Based on Design Choice)
	#TR = 0.8   					# fast whole-brain
	TR  = 2.0   					# moderate whole-brain
	#TR = 2.5   					# conservative whole-brain
     
	# Different Tissue Relaxation times (3T, representative values) (Sec) 
	if tissue == 'Gray_Matter':
		T1 = 1.33 
		T2 = 0.099
     
	if tissue == 'White_Matter':
		T1 = 0.83
		T2 = 0.069
    
	if tissue == 'Arterial_Blood':
		T1 = 1.65
		T2 = 0.165

	if tissue == 'Cer_Spin_Fluid':
		T1 = 4.30  
		T2 = 2.00

  
if B0Mag == 7:
	# Reasonable 7T TR's (Select Based on Design Choice)
	#TR = 1.0   					# fast whole-brain
	TR  = 1.5   					# moderate whole-brain
	#TR = 2.3   					# conservative whole-brain

	# Different Tissue Relaxation times (7T, representative values) (Sec) 
	if tissue == 'Gray_Matter':
		T1 = 1.94 
		T2 = 0.040
     
	if tissue == 'White_Matter':
		T1 = 1.13
		T2 = 0.035
    
	if tissue == 'Arterial_Blood':
		T1 = 2.30
		T2 = 0.065

	if tissue == 'Cer_Spin_Fluid':
		T1 = 4.30  
		T2 = 1.80


if B0Mag == 9.4:
	# Reasonable 9.4T TR's
	#TR = 1.5   					# fast whole-brain
	TR  = 2.5   					# moderate whole-brain
	#TR = 3.5   					# conservative whole-brain
      
	# Different Tissue Relaxation times (9.4T, representative values) (Sec) 
	if tissue == 'Gray_Matter':
		T1 = 1.99 
		T2 = 0.029
     
	if tissue == 'White_Matter':
		T1 =  1.37
		T2 = 0.027
    
	if tissue == 'Arterial_Blood':
		T1 =  2.67
		T2 = 0.053

	if tissue == 'Cer_Spin_Fluid':
		T1 =  4.50 
		T2 = 1.60 

# Ernst Angle for Relevant Tissues
Ernst_alpha =  math.acos(math.exp(-TR/T1))
ninetyalpha = const.pi/2



# Declare Voxel Size (meters)
dx = 1.5e-3 
dy = 1.5e-3
dz = 1.5e-3


# -------------------------------------------------------------- Physical Definitions ------------------------------------------------------------------------ # 

# Define Physical Constants
hbar   = const.hbar                 # Joule*sec
boltzk = const.Boltzmann            # Joule/Kelvin


# Declare Hydrogen Observables
H1gamma = 2*(const.pi)*42.577e6;    # radians/sec/Tesla
H1spin  = 1/2

# Pauli Spin Matricies
sx = np.array([[0,1],[1,0]], complex)
sy = np.array([[0,-1j],[1j,0]], complex)
sz = np.array([[1,0j],[0,-1]], complex)
# Physically Normalized Pauli Mats
Sx = hbar*(1/2)*sx   			    # Joule*sec
Sy = hbar*(1/2)*sy  			    # Joule*sec
Sz = hbar*(1/2)*sz   			    # Joule*sec


# Create the two state spin vector basis
spinup    = np.array([[1],[0]])
spindown  = np.array([[0],[1]])
spinbasis = np.column_stack((spinup, spindown))

# -------------------------------------------------------------- Quantum Mechanics ------------------------------------------------------------------------ # 

# Construct the Zeeman Hamiltonian
H = -H1gamma*(Bx*Sx + By*Sy + Bz*Sz)

# Choose Earnst Angle (choosing tissue)
alpha     = Ernst_alpha 					# Change Based on Design Choice
nH_tissue = nH

# Compute Nuclear Zeeman Splitting
DeltaE = hbar*H1gamma*B0_norm

# Compute Population imbalance (Nupper/Nlower) at room temp (295.5 Kelvin)
popdiff = math.exp(-DeltaE/(boltzk*295.5))

# lower-energy state population
spin_up_alligned  = 1/(1+popdiff)
# higher-energy state population
spin_down_anti = popdiff/(1+popdiff) 

# Thermal Equilibrium Density Matrix
rho_minus = spin_up_alligned*(spinup@spinup.conj().T) + spin_down_anti*(spindown@spindown.conj().T)

# Define B1 via Unitary Quantum Rotation
Ux = expm((-1j*alpha*Sx)/hbar)

# Apply B1 to Ensemble (Rotate our State)
rho_plus = Ux@rho_minus@(Ux.conj().T)


# -------------------------------------------------------------- Construct Evolution SuperOperator ------------------------------------------------------------------------ # 

# Compute dephasing factor
dephaserate = 1/2*(1/T2 - 1/(2*T1))
# Dephasing Operator 
L_phi = sz

# Compute rasing and lowering sum
T1_Rate = 1/T1 						# equals the loweringrate + raisingrate

# We can recover the individual rates from the populationratio
downrate = T1_Rate*1/(1+popdiff)
uprate =  T1_Rate*popdiff/(1+popdiff)

# Raising and Lowering Operators
L_raise = spindown@(spinup).conj().T
L_lower = spinup@(spindown).conj().T

# Create the Superoperator including the Hamiltonian Term and Relaxation
sup_ham = (-1j/hbar)*(np.kron(spinbasis,H) - np.kron(H.T,spinbasis))
sup_phi = dephaserate*(np.kron(L_phi.conj(),L_phi) - 0.5*np.kron(spinbasis,L_phi.conj().T@L_phi) - 0.5*np.kron((L_phi.conj().T@L_phi).T,spinbasis))
sup_up = uprate*(np.kron(L_raise.conj(),L_raise)- 0.5*np.kron(spinbasis,L_raise.conj().T@L_raise) - 0.5*np.kron((L_raise.conj().T@L_raise).T,spinbasis))
sup_down = downrate*(np.kron(L_lower.conj(),L_lower) - 0.5*np.kron(spinbasis,L_lower.conj().T@L_lower) - 0.5*np.kron((L_lower.conj().T@L_lower).T,spinbasis))

# Add effects to Construct full superoperator
Bloch_Operator = sup_ham+sup_phi+sup_up+sup_down


# -------------------------------------------------------------- Evolution of Post-Pulse State ------------------------------------------------------------------------ # 

# Declare introductory animation holds
frame1repeats = 15
frame2repeats = 15
repeatsum = frame1repeats + frame2repeats

# Simulate several multiples of the selected tissue's T2
number_of_T2_multiples = 5.0
num_physical_frames = 870
physt = np.linspace(0.0, number_of_T2_multiples * T2, num_physical_frames)

# Initialize density-matrix history, including artificial hold frames
Rho_T = np.zeros((4, repeatsum + num_physical_frames), dtype=complex)

# Reshape the starting states 
rho_minus_vec = rho_minus.flatten(order='F')
rho_plus_vec = rho_plus.flatten(order='F')


# Fill First few "frames" with initial state (for plotting clarity)
for i in range(frame1repeats):
    Rho_T[:, i] = rho_minus_vec

# Fill Next few in with 2nd frame for clarity
for i in range(frame1repeats, repeatsum):
    Rho_T[:, i] = rho_plus_vec

# Evolve with SuperOperator
for i in tqdm(range(len(physt)), desc="Evolving post-pulse state", unit="frame", dynamic_ncols=True):
    Rho_T[:, i + repeatsum] = expm(Bloch_Operator * physt[i]) @ rho_plus_vec

# Reshape Rho_T back
Rho_T = Rho_T.reshape((2, 2, -1), order="F")

# Number of frames
num_frames = Rho_T.shape[2]

# Initialize expectations
ExpSx_t = np.zeros(num_frames)
ExpSy_t = np.zeros(num_frames)
ExpSz_t = np.zeros(num_frames)

# Now compute expectations for every frame
for k in tqdm(range(num_frames), desc="Computing expectation values", unit="frame", dynamic_ncols=True):
    ExpSx_t[k] = np.real(np.trace(Rho_T[:, :, k] @ Sx))
    ExpSy_t[k] = np.real(np.trace(Rho_T[:, :, k] @ Sy))
    ExpSz_t[k] = np.real(np.trace(Rho_T[:, :, k] @ Sz))

# Compute Bloch Vector by normalizing the averages
rx_t = (2/hbar) * ExpSx_t
ry_t = (2/hbar) * ExpSy_t
rz_t = (2/hbar) * ExpSz_t
R = np.column_stack((rx_t, ry_t, rz_t))


# --------------------------------------------------------- Convert Evolved States to Physical Moments -------------------------------------------------------------- #

# Proton Density of Tissue Voxel
V_voxel = dx*dy*dz;      						# m^3
N_voxel = nH_tissue * V_voxel

# Single Magnetic Moment strengths
mu_x_t = H1gamma * ExpSx_t
mu_y_t = H1gamma * ExpSy_t
mu_z_t = H1gamma * ExpSz_t

# Voxel Magnetic Moments
Mx_t = N_voxel * mu_x_t          
My_t = N_voxel * mu_y_t
Mz_t = N_voxel * mu_z_t
M = np.column_stack((Mx_t, My_t, Mz_t))


# --------------------------------------------------------- Construct Wigner and Husimi Q Representations --------------------------------------------------------- #

# Retain only the physically evolved states and remove artificial introductory hold frames
WignerHusimi_Rho_phys = Rho_T[:, :, repeatsum:]
WignerHusimi_R_phys = R[repeatsum:, :]
WignerHusimi_time = physt

# Number of physical evolution frames
num_WignerHusimi_frames = WignerHusimi_Rho_phys.shape[2]

# Declare angular resolution of the spin phase-space sphere
num_theta_points = 91
num_phi_points = 181
# Polar and azimuthal angular coordinates
WignerHusimi_theta = np.linspace(0.0, const.pi, num_theta_points)
WignerHusimi_phi = np.linspace(0.0, 2.0 * const.pi, num_phi_points)

# Construct spherical angular grid
WignerHusimi_Phi, WignerHusimi_Theta = np.meshgrid(WignerHusimi_phi, WignerHusimi_theta)
# Cartesian components of every unit direction on the sphere
WignerHusimi_sphere_x = np.sin(WignerHusimi_Theta) * np.cos(WignerHusimi_Phi)
WignerHusimi_sphere_y = np.sin(WignerHusimi_Theta) * np.sin(WignerHusimi_Phi)
WignerHusimi_sphere_z = np.cos(WignerHusimi_Theta)

# Two-dimensional identity operator
identity2 = np.eye(2, dtype=complex)

# Preallocate one 2x2 matrix for every point on the spherical grid
WignerHusimi_sigma_dot_n = np.zeros((num_theta_points, num_phi_points, 2, 2), dtype=complex)
# Preallocate one Husimi and Wigner kernels for every spherical direction
Wigner_Kernel = np.zeros((num_theta_points, num_phi_points, 2, 2), dtype=complex)
Husimi_Kernel = np.zeros((num_theta_points, num_phi_points, 2, 2), dtype=complex)
# Construct the spin-direction operator and kernels at every point on the sphere
for theta_index in tqdm(range(num_theta_points), desc="Constructing Wigner and Husimi kernels", unit="theta", dynamic_ncols=True):
    for phi_index in range(num_phi_points):
        # Pull the Cartesian components of the unit vector n
        nx = WignerHusimi_sphere_x[theta_index, phi_index]
        ny = WignerHusimi_sphere_y[theta_index, phi_index]
        nz = WignerHusimi_sphere_z[theta_index, phi_index]

        # Construct the spin-direction operator associated with this sphere point
        spin_direction_operator = nx * sx + ny * sy + nz * sz
        WignerHusimi_sigma_dot_n[theta_index, phi_index, :, :] = spin_direction_operator

        # Construct the Wigner and Husimi Q kernels
        Wigner_Kernel[theta_index, phi_index, :, :] = (identity2 + np.sqrt(3.0) * spin_direction_operator) / (4.0 * const.pi)
        Husimi_Kernel[theta_index, phi_index, :, :] = (identity2 + spin_direction_operator) / (4.0 * const.pi)

# Preallocate the Wigner and Husimi distributions
Wigner = np.zeros((num_WignerHusimi_frames, num_theta_points, num_phi_points))
Husimi = np.zeros((num_WignerHusimi_frames, num_theta_points, num_phi_points))

# Evaluate the distributions for every time and spherical direction
for time_index in tqdm(range(num_WignerHusimi_frames), desc="Constructing Wigner and Husimi distributions", unit="frame", dynamic_ncols=True):
    # Pull the evolved density matrix at the current time
    rho_current = WignerHusimi_Rho_phys[:, :, time_index]

    for theta_index in range(num_theta_points):
        for phi_index in range(num_phi_points):
            # Pull the kernels at the current direction on the sphere
            Wigner_kernel_current = Wigner_Kernel[theta_index, phi_index, :, :]
            Husimi_kernel_current = Husimi_Kernel[theta_index, phi_index, :, :]

            # Evaluate Tr[rho(t) Delta(theta, phi)]
            Wigner[time_index, theta_index, phi_index] = np.real(np.trace(rho_current @ Wigner_kernel_current))
            Husimi[time_index, theta_index, phi_index] = np.real(np.trace(rho_current @ Husimi_kernel_current))

# Store the phase-space histories using single precision to reduce memory
Wigner = Wigner.astype(np.float32)
Husimi = Husimi.astype(np.float32)

# --------------------------------------------------------- Wigner and Husimi Q Plotting Extravaganza -------------------------------------------------------------- #

# Thin the phase-space arrays for interactive plotting
time_plot_stride = 5
theta_plot_stride = 2
phi_plot_stride = 2

# Retain selected physical time frames
WignerHusimi_time_plot = WignerHusimi_time[::time_plot_stride]
# Retain a reduced angular grid for Plotly rendering
WignerHusimi_sphere_x_plot = WignerHusimi_sphere_x[::theta_plot_stride, ::phi_plot_stride]
WignerHusimi_sphere_y_plot = WignerHusimi_sphere_y[::theta_plot_stride, ::phi_plot_stride]
WignerHusimi_sphere_z_plot = WignerHusimi_sphere_z[::theta_plot_stride, ::phi_plot_stride]
# Retain the Wigner and Husimi Q densities in inverse steradians
Wigner_plot = Wigner[::time_plot_stride, ::theta_plot_stride, ::phi_plot_stride]
Husimi_plot = Husimi[::time_plot_stride, ::theta_plot_stride, ::phi_plot_stride]
# Number of phase-space animation frames
num_WignerHusimi_plot_frames = len(WignerHusimi_time_plot)

# Uniform phase-space density associated with the maximally mixed state
uniform_phase_space_density = 1.0 / (4.0 * const.pi)
# Establish fixed color limits centered on the uniform phase-space density
Wigner_color_half_range = np.max(np.abs(Wigner_plot - uniform_phase_space_density))
Husimi_color_half_range = np.max(np.abs(Husimi_plot - uniform_phase_space_density))
# Prevent degenerate color scales if a representation is exactly uniform
if Wigner_color_half_range == 0.0:
    Wigner_color_half_range = 1.0e-12
if Husimi_color_half_range == 0.0:
    Husimi_color_half_range = 1.0e-12
Wigner_color_min = uniform_phase_space_density - Wigner_color_half_range
Wigner_color_max = uniform_phase_space_density + Wigner_color_half_range
Husimi_color_min = uniform_phase_space_density - Husimi_color_half_range
Husimi_color_max = uniform_phase_space_density + Husimi_color_half_range

# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
grid_white = "rgba(255,255,255,0.18)"

# IBM-inspired diverging color scale
phase_space_colorscale = [[0.0, "#0F62FE"], [0.5, "#F5F7FA"], [1.0, "#DA1E28"]]

# Animation controls
frame_duration_ms = 55

# Construct the title shown for a given animation frame
def get_WignerHusimi_frame_title(k):
    current_time = WignerHusimi_time_plot[k]
    current_T2_multiple = current_time / T2
    return f"Spin Phase-Space Density Evolution → t = {current_time:.3e} s = {current_T2_multiple:.2f} T₂"

# Create two three-dimensional spherical panels
fig_WignerHusimi = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_widths=[0.50, 0.50], horizontal_spacing=0.08, subplot_titles=("Wigner Quasiprobability Density", "Husimi Q Probability Density"))

# Trace 0: Wigner quasiprobability density on the unit sphere
fig_WignerHusimi.add_trace(go.Surface(x=WignerHusimi_sphere_x_plot, y=WignerHusimi_sphere_y_plot, z=WignerHusimi_sphere_z_plot, surfacecolor=Wigner_plot[0, :, :], colorscale=phase_space_colorscale, cmin=Wigner_color_min, cmax=Wigner_color_max, showscale=True, colorbar=dict(title=dict(text="W(n,t)<br>sr⁻¹", font=dict(color=text_white)), x=0.455, len=0.78, thickness=16, tickfont=dict(color=text_white), outlinecolor=axis_gold, outlinewidth=1, tickformat=".8f"), lighting=dict(ambient=0.70, diffuse=0.75, roughness=0.85, specular=0.20, fresnel=0.05), lightposition=dict(x=100, y=-100, z=100), hovertemplate="Spin direction<br>nₓ = %{x:.3f}<br>nᵧ = %{y:.3f}<br>n_z = %{z:.3f}<br><br>W(n,t) = %{surfacecolor:.8e} sr⁻¹<extra></extra>", name="Wigner quasiprobability density", showlegend=False), row=1, col=1)

# Trace 1: Husimi Q probability density on the unit sphere
fig_WignerHusimi.add_trace(go.Surface(x=WignerHusimi_sphere_x_plot, y=WignerHusimi_sphere_y_plot, z=WignerHusimi_sphere_z_plot, surfacecolor=Husimi_plot[0, :, :], colorscale=phase_space_colorscale, cmin=Husimi_color_min, cmax=Husimi_color_max, showscale=True, colorbar=dict(title=dict(text="Q(n,t)<br>sr⁻¹", font=dict(color=text_white)), x=1.025, len=0.78, thickness=16, tickfont=dict(color=text_white), outlinecolor=axis_gold, outlinewidth=1, tickformat=".8f"), lighting=dict(ambient=0.70, diffuse=0.75, roughness=0.85, specular=0.20, fresnel=0.05), lightposition=dict(x=100, y=-100, z=100), hovertemplate="Spin direction<br>nₓ = %{x:.3f}<br>nᵧ = %{y:.3f}<br>n_z = %{z:.3f}<br><br>Q(n,t) = %{surfacecolor:.8e} sr⁻¹<extra></extra>", name="Husimi Q probability density", showlegend=False), row=1, col=2)

# Construct animation frames
WignerHusimi_animation_frames = []

for k in tqdm(range(num_WignerHusimi_plot_frames), desc="Building Wigner and Husimi animation frames", unit="frame", dynamic_ncols=True):
    current_frame = go.Frame(name=str(k), data=[go.Surface(surfacecolor=Wigner_plot[k, :, :]), go.Surface(surfacecolor=Husimi_plot[k, :, :])], traces=[0, 1], layout=go.Layout(title=dict(text=get_WignerHusimi_frame_title(k))))
    WignerHusimi_animation_frames.append(current_frame)

print("Building Wigner and Husimi plot environment...")

# Attach all animation frames to the figure
fig_WignerHusimi.frames = WignerHusimi_animation_frames
# Build slider steps
WignerHusimi_slider_steps = []
for k in range(num_WignerHusimi_plot_frames):
    slider_step = dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label="")
    WignerHusimi_slider_steps.append(slider_step)
# Shared scene styling function
def get_phase_space_scene_layout():
    return dict(bgcolor=panel_background, aspectmode="cube", dragmode="orbit", uirevision="constant", camera=dict(eye=dict(x=1.45, y=-1.00, z=0.80)), xaxis=dict(title="nₓ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-1.05, 1.05], showbackground=True), yaxis=dict(title="nᵧ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-1.05, 1.05], showbackground=True), zaxis=dict(title="n_z", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-1.05, 1.05], showbackground=True))
# Style the complete figure
fig_WignerHusimi.update_layout(title=dict(text=get_WignerHusimi_frame_title(0), x=0.5, xanchor="center", font=dict(size=22, color=text_white)), height=760, paper_bgcolor=page_background, plot_bgcolor=panel_background, font=dict(color=text_white), uirevision="constant", margin=dict(l=30, r=90, t=130, b=130))
# Apply identical styling to both spherical scenes
fig_WignerHusimi.update_layout(scene=get_phase_space_scene_layout(), scene2=get_phase_space_scene_layout())
# Style subplot titles
fig_WignerHusimi.update_annotations(font=dict(color=axis_gold, size=17))
# Add play and pause buttons
fig_WignerHusimi.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0.50, y=-0.20, xanchor="center", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, font=dict(color=text_white), buttons=[dict(label="▶ Play", method="animate", args=[None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}]), dict(label="❚❚ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])])])
# Add a draggable frame slider
fig_WignerHusimi.update_layout(sliders=[dict(active=0, x=0.10, len=0.82, y=-0.055, xanchor="left", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, activebgcolor="#0F62FE", tickcolor=axis_gold, font=dict(color=text_white), currentvalue=dict(visible=False), pad=dict(t=30, b=0), steps=WignerHusimi_slider_steps)])
# Add gold borders around both spherical panels
scene1_x0, scene1_x1 = fig_WignerHusimi.layout.scene.domain.x
scene1_y0, scene1_y1 = fig_WignerHusimi.layout.scene.domain.y
scene2_x0, scene2_x1 = fig_WignerHusimi.layout.scene2.domain.x
scene2_y0, scene2_y1 = fig_WignerHusimi.layout.scene2.domain.y
fig_WignerHusimi.add_shape(type="rect", xref="paper", yref="paper", x0=scene1_x0, x1=scene1_x1, y0=scene1_y0, y1=scene1_y1, line=dict(color=axis_gold, width=1), fillcolor="rgba(0,0,0,0)", layer="above")
fig_WignerHusimi.add_shape(type="rect", xref="paper", yref="paper", x0=scene2_x0, x1=scene2_x1, y0=scene2_y0, y1=scene2_y1, line=dict(color=axis_gold, width=1), fillcolor="rgba(0,0,0,0)", layer="above")

print("Writing interactive Wigner and Husimi HTML file...")

# Open the interactive animation in the default web browser
fig_WignerHusimi.write_html("Lindbladian_Wigner_Husimi_T2_Simulator.html", auto_open=True, auto_play=False, config={"scrollZoom": True, "displaylogo": False, "responsive": True}, post_script="document.title = 'Lindbladian Wigner and Husimi T2 Simulator';")

print("Wigner and Husimi HTML file complete.")
