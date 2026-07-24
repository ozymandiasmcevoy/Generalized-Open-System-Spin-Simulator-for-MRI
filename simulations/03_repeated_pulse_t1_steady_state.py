
# This script constructs a tissue-configurable, quasi-classical, physics-based simulator of voxel-scale nuclear magnetic resonance Bloch dynamics. (Ozymandias McEvoy)
# RF excitation is modeled as a sequence of ideal, instantaneous, fixed-flip-angle unitary rotations. 
# Longitudinal and transverse relaxation are modeled through idealized T1 and T2 relaxation channels.
# The script simulates voxel-level magnetization dynamics across multiple repetition times for a selected tissue.
# Repeated pulses are simulated to demonstrate incomplete longitudinal recovery, progressive saturation, and the emergence of a flip-angle-dependent steady state.


# Import Relevent Packages
import scipy.constants as const
from scipy.linalg import expm
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


#This script has a memory concern, github repos require file size <1mb so keep pulse_size/plot_stride reletively stable for repo HTML # (RN all is well)



# ------------------------------------------------------------ Experimental Knobs and Biological Constants------------------------------------------------------------ #

# Relaxation Times, TRs, and TEs were estimated from the literature to be physically reasonable to allow for realistic simulation. This section can be easily modified #
# to accurately reflect specific experimental conditions 																											   #



# Physical Proton densities [protons / m^3]
nH2O_ref      = 6.666e28;   	   # pure-water reference at ~25 C

# Make Tissue Choice
tissue = 'Gray_Matter'
#tissue = 'White_Matter'
#tissue = 'Arterial_Blood'
#tissue = 'Cer_Spin_Fluid'

if tissue == 'Gray_Matter':
    PD = 0.84
    nH = PD * nH2O_ref
     
if tissue == 'White_Matter':
    PD = 0.69
    nH = PD * nH2O_ref
     
if tissue == 'Arterial_Blood':
    PD = 0.83
    nH = PD * nH2O_ref
    
if tissue == 'Cer_Spin_Fluid':
    PD = 1.00
    nH = PD * nH2O_ref
    

# Relevent Field Strengths (Select Based on Design Choice)
#B0Mag = 3                          # Tesla
B0Mag  = 7                          # Tesla
#B0Mag = 9.4                          # Tesla 


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
    

	if tissue == 'Gray_Matter':
		T1 = 1.99 
		T2 = 0.029
        
	if tissue == 'White_Matter':
		T1 = 1.37 
		T2 = 0.027
        
	if tissue == 'Arterial_Blood':
		T1 = 2.67 
		T2 = 0.053
        
	if tissue == 'Cer_Spin_Fluid':
		T1 = 4.50 
		T2 = 1.60

# Earnst Angle for Relevant Tissues
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

# Number of times to repeat pulse
number_of_pulses = 6

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



# -------------------------------------------------------------- Evolution Across Repeated Pulses ------------------------------------------------------------------------ #

# Number of physical samples stored during each repetition time
SimPointsPerPulse = 4000
# Total number of physical samples
Total_Sim_Points = number_of_pulses * SimPointsPerPulse
# Local physical time following each pulse
perpulse_Phys_Time = np.linspace(0.0, TR, SimPointsPerPulse, endpoint=False)

# Time spacing between stored samples
dt = TR / SimPointsPerPulse

# Initialize density-matrix and global-time histories
Rho_sim = np.zeros((2, 2, Total_Sim_Points), dtype=complex)
t_sim = np.zeros(Total_Sim_Points)

# State immediately after the first pulse
rho_plus = Ux @ rho_minus @ Ux.conj().T

# Precompute the SuperOperator for every local time within one TR
perpulse_propagators = np.zeros((SimPointsPerPulse, 4, 4), dtype=complex)
for g in tqdm(range(SimPointsPerPulse), desc="Precomputing propagators"):
    perpulse_propagators[g] = expm(Bloch_Operator * perpulse_Phys_Time[g])

# Propagator used to advance exactly one complete TR
TR_propagator = expm(Bloch_Operator * TR)

# The first pulse begins from thermal equilibrium
rho_before_pulse = rho_minus.copy()

# Simulate each pulse block and subsequent relaxation block
for pulse_idx in tqdm(range(number_of_pulses), desc="Simulating pulse blocks"):

    # Apply the instantaneous RF pulse
    rho_after_pulse = Ux @ rho_before_pulse @ Ux.conj().T
    # Vectorize using MATLAB-compatible column ordering
    rho_after_pulse_vec = rho_after_pulse.flatten(order="F")
    # Starting storage index for this pulse block
    block_start = pulse_idx * SimPointsPerPulse

    # Evolve through the current repetition time
    for g in range(SimPointsPerPulse):
        idx = block_start + g
        tau = perpulse_Phys_Time[g]
        rho_tau_vec = perpulse_propagators[g] @ rho_after_pulse_vec
        Rho_sim[:, :, idx] = rho_tau_vec.reshape((2, 2), order="F")
        t_sim[idx] = pulse_idx * TR + tau

    # Set last frame as first frame of next pulse 
    rho_before_next_pulse_vec = TR_propagator @ rho_after_pulse_vec
    rho_before_pulse = rho_before_next_pulse_vec.reshape((2, 2), order="F")
    
# -------------------------------------------------------------- Compute Expectation Values ------------------------------------------------------------------------ #

# Number of physical simulation frames
num_frames = Rho_sim.shape[2]

# Initialize expectation-value arrays
ExpSx_t = np.zeros(num_frames)
ExpSy_t = np.zeros(num_frames)
ExpSz_t = np.zeros(num_frames)

# Compute spin expectation values at every physical time point
for k in range(num_frames):
    ExpSx_t[k] = np.real(np.trace(Rho_sim[:, :, k] @ Sx))
    ExpSy_t[k] = np.real(np.trace(Rho_sim[:, :, k] @ Sy))
    ExpSz_t[k] = np.real(np.trace(Rho_sim[:, :, k] @ Sz))

# Normalize the spin expectations to obtain the Bloch-vector coordinates
rx_t = (2.0 / hbar) * ExpSx_t
ry_t = (2.0 / hbar) * ExpSy_t
rz_t = (2.0 / hbar) * ExpSz_t
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

# Lightly thin the dense physical simulation for animation efficiency
plot_stride = 160
circle_stride = 1

Mx_plot = Mx_t[::plot_stride]
My_plot = My_t[::plot_stride]
Mz_plot = Mz_t[::plot_stride]
t_plot = t_sim[::plot_stride]

# Introductory animation holds
frame1repeats = 10
frame2repeats = 10
repeatsum = frame1repeats + frame2repeats

# Magnetization before the first pulse
Mx_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sx))
My_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sy))
Mz_minus = N_voxel * H1gamma * np.real(np.trace(rho_minus @ Sz))

# Magnetization immediately after the first pulse
Mx_plus = N_voxel * H1gamma * np.real(np.trace(rho_plus @ Sx))
My_plus = N_voxel * H1gamma * np.real(np.trace(rho_plus @ Sy))
Mz_plus = N_voxel * H1gamma * np.real(np.trace(rho_plus @ Sz))

# Attach the artificial introductory holds to the real physical evolution
Mx_final = np.concatenate((np.full(frame1repeats, Mx_minus), np.full(frame2repeats, Mx_plus), Mx_plot))
My_final = np.concatenate((np.full(frame1repeats, My_minus), np.full(frame2repeats, My_plus), My_plot))
Mz_final = np.concatenate((np.full(frame1repeats, Mz_minus), np.full(frame2repeats, Mz_plus), Mz_plot))

t_final = np.concatenate((np.zeros(repeatsum), t_plot))
M_final = np.column_stack((Mx_final, My_final, Mz_final))

# Transverse magnitude
r_perp_final = np.sqrt(Mx_final**2 + My_final**2)

# Number of animated physical frames corresponding to one TR
frames_per_pulse_plot = int(np.ceil(SimPointsPerPulse / plot_stride))

# Transverse magnitude during the thinned physical evolution
r_perp_plot = np.sqrt(Mx_plot**2 + My_plot**2)
# Exact post-pulse peak from the first frame of every pulse block
pulse_peak_indices = np.arange(number_of_pulses) * SimPointsPerPulse
pulse_peak_times = t_sim[pulse_peak_indices]
pulse_peak_values = np.sqrt(Mx_t[pulse_peak_indices]**2 + My_t[pulse_peak_indices]**2)
# Vertical limit for the transverse-magnetization panel
right_y_max = 1.18 * np.max(r_perp_plot)
if right_y_max == 0:
    right_y_max = 1.0

# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
vector_red = "#FF4D4D"
circle_white = "#FFFFFF"
grid_white = "rgba(255,255,255,0.18)"

# One persistent trail color for each pulse block
pulse_colors = ["#4DDFFF", "#39FF14", "#FFCC00", "#FF40FF", "#FF8C42", "#B388FF", "#00FFB3"]

# Animation controls
frame_duration_ms = 55

# Number of points used to draw each transverse circle
theta = np.linspace(0.0, 2.0 * np.pi, 120)

# Number of animation frames
num_plot_frames = len(t_final)

# Symmetric transverse-axis limits
xy_limit = 1.10 * np.max(np.abs(np.concatenate((Mx_final, My_final))))
if xy_limit == 0:
    xy_limit = 1.0

# Longitudinal-axis limits
z_lower = min(0.0, 1.05 * np.min(Mz_final))
z_upper = 1.05 * np.max(Mz_final)
if z_upper == z_lower:
    z_upper = z_lower + 1.0

# Construct the title shown for each animation frame
def get_frame_title(k):
    if k < frame1repeats:
        return f"Equilibrium state → hold frame {k + 1} of {frame1repeats}"
    if k < repeatsum:
        return f"Immediately after first pulse → hold frame {k - frame1repeats + 1} of {frame2repeats}"
    physical_frame = k - repeatsum
    pulse_index = min(physical_frame // frames_per_pulse_plot, number_of_pulses - 1)
    return f"Pulse {pulse_index + 1} of {number_of_pulses} → t = {t_final[k]:.3e} s"

# Construct the transverse circle at the current longitudinal position
def get_circle_coordinates(k):
    current_radius = r_perp_final[k]
    circle_x = current_radius * np.cos(theta)
    circle_y = current_radius * np.sin(theta)
    circle_z = np.full(theta.shape, Mz_final[k])

    return circle_x, circle_y, circle_z

# Create one three-dimensional panel and one transverse-magnitude panel
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[0.57, 0.43], horizontal_spacing=0.07, subplot_titles=("Bloch Magnetization", "Repeated-Pulse Transverse Magnetization"))
# Build the initial transverse circle
circle_x_initial, circle_y_initial, circle_z_initial = get_circle_coordinates(0)
# Trace 0: current magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0, Mx_final[0]], y=[0.0, My_final[0]], z=[0.0, Mz_final[0]], mode="lines+markers", line=dict(color=vector_red, width=8), marker=dict(size=[2, 7], color=vector_red), name="Magnetization vector", showlegend=False, hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"), row=1, col=1)
# Trace 1: current transverse circle
fig.add_trace(go.Scatter3d(x=circle_x_initial, y=circle_y_initial, z=circle_z_initial, mode="lines", line=dict(color=circle_white, width=6), name="Current transverse circle", showlegend=False, hoverinfo="skip"), row=1, col=1)
# One accumulated circle-history trace for each pulse
pulse_history_trace_indices = []

for pulse_index in range(number_of_pulses):
    pulse_color = pulse_colors[pulse_index % len(pulse_colors)]
    pulse_history_trace_indices.append(len(fig.data))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", line=dict(color=pulse_color, width=3), name=f"Pulse {pulse_index + 1} history", showlegend=False, hoverinfo="skip"), row=1, col=1)

# Complete transverse-magnitude time series
fig.add_trace(go.Scatter(x=t_plot, y=r_perp_plot, mode="lines", line=dict(color="#39FF14", width=3), name="M⊥(t)", hovertemplate="Time = %{x:.4f} s<br>M⊥ = %{y:.3e}<extra></extra>"), row=1, col=2)
# Exact post-pulse peak markers
peak_labels = [f"P{pulse_index + 1}" for pulse_index in range(number_of_pulses)]
fig.add_trace(go.Scatter(x=pulse_peak_times, y=pulse_peak_values, mode="markers+text", marker=dict(color=axis_gold, size=11, line=dict(color=text_white, width=1)), text=peak_labels, textposition="top center", textfont=dict(color=text_white, size=12), cliponaxis=False, name="Post-pulse peaks", hovertemplate="Pulse %{text}<br>Time = %{x:.4f} s<br>Peak M⊥ = %{y:.3e}<extra></extra>"), row=1, col=2)
# Moving point on the transverse-magnitude curve
fig.add_trace(go.Scatter(x=[t_final[0]], y=[r_perp_final[0]], mode="markers", marker=dict(color="#FF40FF", size=12, line=dict(color=text_white, width=1)), name="Current state", showlegend=False, hovertemplate="Time = %{x:.4f} s<br>M⊥ = %{y:.3e}<extra></extra>"), row=1, col=2)
moving_point_trace_index = len(fig.data) - 1
# Independent historical-circle storage for every pulse
trail_x = [[] for _ in range(number_of_pulses)]
trail_y = [[] for _ in range(number_of_pulses)]
trail_z = [[] for _ in range(number_of_pulses)]

# Traces changed during the animation
animated_trace_indices = [0, 1] + pulse_history_trace_indices + [moving_point_trace_index]

# Construct all animation frames
animation_frames = []

for k in tqdm(range(num_plot_frames), desc="Building animation frames"):
    circle_x, circle_y, circle_z = get_circle_coordinates(k)

    # Accumulate circles during the physical repeated-pulse simulation
    if k >= repeatsum:
        physical_frame = k - repeatsum
        pulse_index = min(physical_frame // frames_per_pulse_plot, number_of_pulses - 1)
        frame_within_pulse = physical_frame % frames_per_pulse_plot

        if frame_within_pulse % circle_stride == 0:
            trail_x[pulse_index].extend(circle_x.tolist())
            trail_x[pulse_index].append(None)
            trail_y[pulse_index].extend(circle_y.tolist())
            trail_y[pulse_index].append(None)
            trail_z[pulse_index].extend(circle_z.tolist())
            trail_z[pulse_index].append(None)

    # Current magnetization vector
    frame_data = [go.Scatter3d(x=[0.0, Mx_final[k]], y=[0.0, My_final[k]], z=[0.0, Mz_final[k]])]
    # Current transverse circle
    frame_data.append(go.Scatter3d(x=circle_x, y=circle_y, z=circle_z))

    # Accumulated histories for all pulse blocks
    for pulse_index in range(number_of_pulses):
        if len(trail_x[pulse_index]) == 0:
            frame_trail_x = [None]
            frame_trail_y = [None]
            frame_trail_z = [None]
        else:
            frame_trail_x = trail_x[pulse_index].copy()
            frame_trail_y = trail_y[pulse_index].copy()
            frame_trail_z = trail_z[pulse_index].copy()
        frame_data.append(go.Scatter3d(x=frame_trail_x, y=frame_trail_y, z=frame_trail_z))
    # Moving point on the transverse-magnitude curve
    frame_data.append(go.Scatter(x=[t_final[k]], y=[r_perp_final[k]]))
    current_frame = go.Frame(name=str(k), data=frame_data, traces=animated_trace_indices, layout=go.Layout(title=dict(text=get_frame_title(k))))
    animation_frames.append(current_frame)

print("Building plot environment...")

# Attach the animation frames
fig.frames = animation_frames
# Construct slider steps
slider_steps = []
for k in range(num_plot_frames):
    slider_step = dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label="")
    slider_steps.append(slider_step)
# Style the complete figure
fig.update_layout(title=dict(text=get_frame_title(0), x=0.5, xanchor="center", font=dict(size=22, color=text_white)), width=1450, height=800, paper_bgcolor=page_background, plot_bgcolor=panel_background, font=dict(color=text_white), uirevision="constant", margin=dict(l=40, r=40, t=105, b=180))
# Style the three-dimensional panel
fig.update_layout(scene=dict(bgcolor=panel_background, aspectmode="cube", dragmode="orbit", uirevision="constant", camera=dict(eye=dict(x=1.45, y=-1.0, z=0.8)), xaxis=dict(title="Mₓ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-xy_limit, xy_limit], tickformat=".1e", showbackground=True), yaxis=dict(title="Mᵧ", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[-xy_limit, xy_limit], tickformat=".1e", showbackground=True), zaxis=dict(title="M_z", color=axis_gold, backgroundcolor=panel_background, gridcolor=grid_white, zerolinecolor=axis_gold, linecolor=axis_gold, range=[z_lower, z_upper], tickformat=".1e", showbackground=True)))
# Style the transverse-magnitude panel
fig.update_layout(xaxis=dict(title="Physical time (s)", color=axis_gold, linecolor=axis_gold, gridcolor=grid_white, zeroline=False, showline=True, mirror=True, range=[0.0, number_of_pulses * TR]), yaxis=dict(title="Transverse magnetization, M⊥", color=axis_gold, linecolor=axis_gold, gridcolor=grid_white, zeroline=False, showline=True, mirror=True, tickformat=".2e", range=[0.0, right_y_max]))
# Mark the boundaries between pulse blocks
for pulse_index in range(1, number_of_pulses):
    pulse_time = pulse_index * TR
    fig.add_shape(type="line", x0=pulse_time, x1=pulse_time, y0=0.0, y1=1.0, xref="x", yref="y domain", line=dict(color=axis_gold, width=1.2, dash="dot"), opacity=0.55, layer="below")
# Style subplot titles
fig.update_annotations(font=dict(color=axis_gold, size=17))
# Position the compact legend inside the right panel
fig.update_layout(legend=dict(x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(6,24,43,0.88)", bordercolor=axis_gold, borderwidth=1, font=dict(color=text_white)))
# Add play and pause buttons
fig.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0.43, y=-0.19, xanchor="center", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, font=dict(color=text_white), buttons=[dict(label="▶ Play", method="animate", args=[None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}]), dict(label="❚❚ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])])])
# Add the draggable frame slider
fig.update_layout(sliders=[dict(active=0, x=0.10, len=0.82, y=-0.055, xanchor="left", yanchor="top", bgcolor=panel_background, bordercolor=axis_gold, activebgcolor=pulse_colors[0], tickcolor=axis_gold, ticklen=0, minorticklen=0, tickwidth=0, font=dict(color=text_white), currentvalue=dict(visible=False), pad=dict(t=30, b=0), steps=slider_steps)])
# Add a gold border around the three-dimensional panel
scene_x0, scene_x1 = fig.layout.scene.domain.x
scene_y0, scene_y1 = fig.layout.scene.domain.y
fig.add_shape(type="rect", xref="paper", yref="paper", x0=scene_x0, x1=scene_x1, y0=scene_y0, y1=scene_y1, line=dict(color=axis_gold, width=1.5), fillcolor="rgba(0,0,0,0)", layer="above")

print("Writing interactive HTML file...")

# Save and open the standalone simulator
fig.write_html("Lindbladian_Bloch_T1_Steady_State_Simulator.html", auto_open=True, auto_play=False, config={"scrollZoom": True, "displaylogo": False, "responsive": True}, post_script="document.title = 'Lindbladian Bloch T1 and Steady-State Simulator';")

print("HTML file complete.")