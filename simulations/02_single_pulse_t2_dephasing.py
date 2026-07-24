
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


# --------------------------------------------------------- Plotting Extravaganza -------------------------------------------------------------- #


# Build intro-hold states explicitly
Mx_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sx))
My_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sy))
Mz_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sz))
Mx_plus  = N_voxel * H1gamma * np.real(np.trace(rho_plus  @ Sx))
My_plus  = N_voxel * H1gamma * np.real(np.trace(rho_plus  @ Sy))
Mz_plus  = N_voxel * H1gamma * np.real(np.trace(rho_plus  @ Sz))

# Physical evolution after the introductory hold frames
Mx_phys = Mx_t[repeatsum:]
My_phys = My_t[repeatsum:]
Mz_phys = Mz_t[repeatsum:]
#t_phys  = t[repeatsum:]
t_phys  = physt

# Thin arrays for plotting
plot_stride = 5
circle_stride = 8
Mx_plot = Mx_phys[::plot_stride]
My_plot = My_phys[::plot_stride]
Mz_plot = Mz_phys[::plot_stride]
t_plot = t_phys[::plot_stride]

# Rebuild arrays with introductory hold frames
Mx_final = np.concatenate((np.full(frame1repeats, Mx_minus),np.full(frame2repeats, Mx_plus), Mx_plot))
My_final = np.concatenate((np.full(frame1repeats, My_minus),np.full(frame2repeats, My_plus),My_plot))
Mz_final = np.concatenate((np.full(frame1repeats, Mz_minus),np.full(frame2repeats, Mz_plus),Mz_plot))
t_final = np.concatenate((np.zeros(repeatsum),t_plot))

M_final = np.column_stack((Mx_final, My_final, Mz_final))

# Transverse magnitude
r_perp_final = np.sqrt(Mx_final**2 + My_final**2)

# Reference at start of physical evolution
rperp_ref = r_perp_final[repeatsum]
rperp_37 = np.exp(-1) * rperp_ref


# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
vector_red = "#FF4D4D"
circle_white = "#FFFFFF"
trail_cyan = "#4DDFFF"
decay_green = "#39FF14"
reference_red = "#FF6B6B"
current_magenta = "#FF40FF"
grid_white = "rgba(255,255,255,0.18)"

# Animation controls
frame_duration_ms = 55

# Number of points used to draw each transverse circle
theta = np.linspace(0.0, 2.0 * np.pi, 120)

# Number of frames in the thinned plotting arrays
num_plot_frames = len(t_final)

# Equal data-driven limits for the three-dimensional scene
max_abs_magnetization = np.max(np.abs(M_final))

if max_abs_magnetization == 0:
    axis_limit = 1.0
else:
    axis_limit = 1.10 * max_abs_magnetization

# Vertical-axis limit for the transverse-decay panel
right_y_max = 1.10 * np.max(r_perp_final)

if right_y_max == 0:
    right_y_max = 1.0

# First physical state occurs immediately after the introductory frames
free_evolution_start = t_final[repeatsum]

# Construct the title shown for a given animation frame
def get_frame_title(k):
    if k < frame1repeats:
        return f"Equilibrium state → hold frame {k + 1} of {frame1repeats}"

    if k < repeatsum:
        return f"Immediately after pulse → hold frame {k - frame1repeats + 1} of {frame2repeats}"

    return f"Post-pulse evolution → t = {t_final[k]:.3e} s"

# Construct the transverse circle at the current z height
def get_circle_coordinates(k):
    current_radius = r_perp_final[k]
    circle_x = current_radius * np.cos(theta)
    circle_y = current_radius * np.sin(theta)
    circle_z = np.full(theta.shape, Mz_final[k])

    return circle_x, circle_y, circle_z

# Create one three-dimensional panel and one conventional Cartesian panel
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[0.56, 0.44], horizontal_spacing=0.08, subplot_titles=("Bloch Magnetization", "Transverse Decay"))
# Build the first transverse circle
circle_x_initial, circle_y_initial, circle_z_initial = get_circle_coordinates(0)
# Trace 0: current magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0, Mx_final[0]], y=[0.0, My_final[0]], z=[0.0, Mz_final[0]], mode="lines+markers", line=dict(color=vector_red, width=8), marker=dict(size=[2, 6], color=vector_red), name="Magnetization vector", showlegend=False, hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"), row=1, col=1)
# Trace 1: current transverse-radius circle
fig.add_trace(go.Scatter3d(x=circle_x_initial, y=circle_y_initial, z=circle_z_initial, mode="lines", line=dict(color=circle_white, width=6), name="Current transverse circle", showlegend=False, hoverinfo="skip"), row=1, col=1)
# Trace 2: accumulated transverse-circle history
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color=trail_cyan, width=2), name="Circle history", showlegend=False, hoverinfo="skip"), row=1, col=1)
# Trace 3: complete transverse-magnitude curve
fig.add_trace(go.Scatter(x=t_final, y=r_perp_final, mode="lines", line=dict(color=decay_green, width=3), name="M⊥(t)", hovertemplate="Time = %{x:.4f} s<br>M⊥ = %{y:.3e}<extra></extra>"), row=1, col=2)
# Trace 4: 37-percent reference line
fig.add_trace(go.Scatter(x=[np.min(t_final), np.max(t_final)], y=[rperp_37, rperp_37], mode="lines", line=dict(color=reference_red, width=2, dash="dash"), name="0.37 × M⊥(0)", hoverinfo="skip"), row=1, col=2)
# Trace 5: start-of-free-evolution reference line
fig.add_trace(go.Scatter(x=[free_evolution_start, free_evolution_start], y=[0.0, right_y_max], mode="lines", line=dict(color=text_white, width=2, dash="dot"), name="Start of free evolution", hoverinfo="skip"), row=1, col=2)
# Trace 6: moving point on the transverse-decay curve
fig.add_trace(go.Scatter(x=[t_final[0]], y=[r_perp_final[0]], mode="markers", marker=dict(color=current_magenta, size=12, line=dict(color="white", width=1)), name="Current point", hovertemplate="Time = %{x:.4f} s<br>M⊥ = %{y:.3e}<extra></extra>"), row=1, col=2)

# Empty lists used to accumulate historical circles
trail_x = []
trail_y = []
trail_z = []

# Construct the animation frames
animation_frames = []

for k in tqdm(range(num_plot_frames), desc="Building animation frames", unit="frame", dynamic_ncols=True):
    circle_x, circle_y, circle_z = get_circle_coordinates(k)

    # Begin adding historical circles only during physical evolution
    if k >= repeatsum:
        relative_frame = k - repeatsum

        if relative_frame % circle_stride == 0:
            trail_x.extend(circle_x.tolist())
            trail_x.append(None)

            trail_y.extend(circle_y.tolist())
            trail_y.append(None)

            trail_z.extend(circle_z.tolist())
            trail_z.append(None)

    # None provides an initially empty Plotly trace
    if len(trail_x) == 0:
        frame_trail_x = [None]
        frame_trail_y = [None]
        frame_trail_z = [None]
    else:
        frame_trail_x = trail_x.copy()
        frame_trail_y = trail_y.copy()
        frame_trail_z = trail_z.copy()

    # Only traces 0, 1, 2, and 6 change during the animation
    current_frame = go.Frame(name=str(k), data=[go.Scatter3d(x=[0.0, Mx_final[k]], y=[0.0, My_final[k]], z=[0.0, Mz_final[k]]), go.Scatter3d(x=circle_x, y=circle_y, z=circle_z), go.Scatter3d(x=frame_trail_x, y=frame_trail_y, z=frame_trail_z), go.Scatter(x=[t_final[k]], y=[r_perp_final[k]])], traces=[0, 1, 2, 6], layout=go.Layout(title=dict(text=get_frame_title(k))))
    animation_frames.append(current_frame)

print("Building Plot Environment...")

# Attach all animation frames to the figure
fig.frames = animation_frames

# Build slider steps
slider_steps = []

for k in range(num_plot_frames):
    slider_step = dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label="")
    slider_steps.append(slider_step)

# Style the complete figure
fig.update_layout(title=dict(text=get_frame_title(0), x=0.5, xanchor="center", font=dict(size=22, color=text_white)), height=760, paper_bgcolor=page_background, plot_bgcolor=panel_background, font=dict(color=text_white), uirevision="constant", margin=dict(l=30, r=30, t=105, b=130))
# Style the three-dimensional scene
fig.update_layout(scene=dict(bgcolor=panel_background, aspectmode="cube", dragmode="orbit", uirevision="constant", camera=dict(eye=dict(x=1.45, y=-1.0, z=0.8)), xaxis=dict(title="Mₓ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-axis_limit, axis_limit], tickformat=".1e", showbackground=True), yaxis=dict(title="Mᵧ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-axis_limit, axis_limit], tickformat=".1e", showbackground=True), zaxis=dict(title="M_z", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-axis_limit, axis_limit], tickformat=".1e", showbackground=True)))
# Style the transverse-decay axes
fig.update_layout(xaxis=dict(title="Time (s)", color=axis_gold, linecolor=axis_gold, gridcolor=grid_white, zeroline=False, showline=True, mirror=True), yaxis=dict(title="Transverse magnetization, M⊥", color=axis_gold, linecolor=axis_gold, gridcolor=grid_white, zeroline=False, showline=True, mirror=True, tickformat=".2e", range=[0.0, right_y_max]))
# Style subplot titles
fig.update_annotations(font=dict(color=axis_gold, size=17))
# Position the legend within the right-hand plot
fig.update_layout(legend=dict(x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(6,24,43,0.88)", bordercolor=axis_gold, borderwidth=1, font=dict(color=text_white)))
# Add play and pause buttons
fig.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0.43, y=-0.20, xanchor="center", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, font=dict(color=text_white), buttons=[dict(label="▶ Play", method="animate", args=[None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}]), dict(label="❚❚ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])])])
# Add a draggable frame slider without hundreds of overlapping labels
fig.update_layout(sliders=[dict(active=0, x=0.10, len=0.82, y=-0.055, xanchor="left", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, activebgcolor=decay_green, tickcolor=axis_gold, font=dict(color=text_white), currentvalue=dict(visible=False), pad=dict(t=30, b=0), steps=slider_steps)])


# Add gold border around the three-dimensional Bloch panel
scene_x0, scene_x1 = fig.layout.scene.domain.x
scene_y0, scene_y1 = fig.layout.scene.domain.y
fig.add_shape(
    type="rect",
    xref="paper",
    yref="paper",
    x0=scene_x0,
    x1=scene_x1,
    y0=scene_y0,
    y1=scene_y1,
    line=dict(color=axis_gold, width=1),
    fillcolor="rgba(0,0,0,0)",
    layer="above")

print("Writing interactive HTML file...")

# Open the interactive animation in the default web browser
fig.write_html("Lindbladian_Bloch_T2_Simulator.html", auto_open=True, auto_play=False, config={"scrollZoom": True, "displaylogo": False, "responsive": True}, post_script="document.title = 'Lindbladian Bloch T2 Simulator';")

print("HTML file complete.")