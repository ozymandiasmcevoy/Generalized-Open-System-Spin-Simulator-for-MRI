# The Below Script will be modified to show two evolution trajectories... one with a stimulated echo and one without....... on the right it will show an EPG


# Import Relevent Packages
import scipy.constants as const
from scipy.linalg import expm
from scipy.stats import norm
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
#B0Mag = 9.4                        # Tesla 

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

# Number of discrete off-resonance frequency groups
num_isochromats = 41
# Standard deviation of the Gaussian frequency-offset distribution
offset_sd_hz = 8.0
# Divide the Gaussian distribution into equal-probability bins
probability_edges = np.linspace(0.0,1.0,num_isochromats+1)
# Select the probability midpoint of each bin
probability_midpoints = 0.5*(probability_edges[:-1]+probability_edges[1:])
# Convert Gaussian probability positions into frequency offsets
offset_hz = offset_sd_hz*norm.ppf(probability_midpoints)
# Convert frequency offsets from Hz to angular frequency in rad/s
delta_omega = 2*np.pi*offset_hz

# -------------------------------------------------------------- Quantum Mechanics ------------------------------------------------------------------------ # 

# Construct the single pure Zeeman Hamiltonian
H = -H1gamma*(Bx*Sx + By*Sy + Bz*Sz)


# Choose Angle
#alpha = Ernst_alpha
alpha = ninetyalpha
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

# Preallocate isochromat-specific Hamiltonians and Hamiltonian superoperators
H_iso = np.zeros((num_isochromats,2,2),dtype=complex)
sup_ham_iso = np.zeros((num_isochromats,4,4),dtype=complex)
# Compute The hamiltonian and its superop from the pure hamiltonian for each
for j in range(num_isochromats):
    H_iso[j] = H-delta_omega[j]*Sz
    sup_ham_iso[j] = (-1j/hbar)*(np.kron(spinbasis,H_iso[j])-np.kron(H_iso[j].T,spinbasis))


# Compute rasing and lowering sum
T1_Rate = 1/T1 						# equals the loweringrate + raisingrate
# We can recover the individual rates from the populationratio
downrate = T1_Rate*1/(1+popdiff)
uprate =  T1_Rate*popdiff/(1+popdiff)
# Raising and Lowering Operators
L_raise = spindown@(spinup).conj().T
L_lower = spinup@(spindown).conj().T

# Create the Superoperator including the Hamiltonian Term and Relaxation
sup_phi = dephaserate*(np.kron(L_phi.conj(),L_phi) - 0.5*np.kron(spinbasis,L_phi.conj().T@L_phi) - 0.5*np.kron((L_phi.conj().T@L_phi).T,spinbasis))
sup_up = uprate*(np.kron(L_raise.conj(),L_raise)- 0.5*np.kron(spinbasis,L_raise.conj().T@L_raise) - 0.5*np.kron((L_raise.conj().T@L_raise).T,spinbasis))
sup_down = downrate*(np.kron(L_lower.conj(),L_lower) - 0.5*np.kron(spinbasis,L_lower.conj().T@L_lower) - 0.5*np.kron((L_lower.conj().T@L_lower).T,spinbasis))

# Add effects to Construct full superoperator
Bloch_Operator_iso = sup_ham_iso+sup_phi+sup_up+sup_down


# --------------------------------------------------------------Initial Evolution of State ------------------------------------------------------------------------ # 

# Declare introductory animation holds
frame1repeats = 15
frame2repeats = 15
repeatsum = frame1repeats + frame2repeats

# Declare Echo Time (T2 as representative choice)
TE = T2
time_for_pulse2 = TE/2
mixing_time = 1.5*time_for_pulse2
time_for_pulse3 = time_for_pulse2+mixing_time
simulation_end = 2*TE

# Establish the physical sampling interval from the pre-pulse segment
num_pre_frames = 236
physt = np.linspace(0.0,time_for_pulse2,num_pre_frames)
sample_dt = physt[1]-physt[0]

# Automatically preserve that sampling interval after the 90-degree pulse
post_duration = simulation_end-time_for_pulse2
num_post_frames = int(round(post_duration/sample_dt))+1

# Initialize density-matrix history, including artificial hold frames
Pre_Rho_T = np.zeros((num_isochromats, 4, repeatsum + num_pre_frames), dtype=complex)
# Reshape the starting states 
rho_minus_vec = rho_minus.flatten(order='F')
rho_plus_vec = rho_plus.flatten(order='F')

# Fill First few "frames" with initial state (for plotting clarity)
for i in range(frame1repeats):
    Pre_Rho_T[:,:,i] = rho_minus_vec

# Fill Next few in with 2nd frame for clarity
for i in range(frame1repeats, repeatsum):
    Pre_Rho_T[:,:,i] = rho_plus_vec

# Evolve every isochromat from excitation to TE/2
for j in tqdm(range(num_isochromats),desc="Evolving isochromats after 1st pulse"):
    for i in range(num_pre_frames):
        Pre_Rho_T[j,:,i+repeatsum] = expm(Bloch_Operator_iso[j]*physt[i])@rho_plus_vec

# Obtain every isochromat's state immediately before the 90-degree pulse
rho_tau_vec = Pre_Rho_T[:,:,-1]

# Pulse the midpoint state and reflatten all
rho_SE_vec = np.zeros((num_isochromats,4),dtype=complex)
for j in range(num_isochromats):
    rho_tau_j = rho_tau_vec[j].reshape((2,2),order="F")
    rho_SE_j = Ux@rho_tau_j@Ux.conj().T
    rho_SE_vec[j] = rho_SE_j.flatten(order="F")

# Establish Length of new time evolution
t_post_local = np.linspace(0.0,simulation_end-time_for_pulse2,num_post_frames)
t_post = time_for_pulse2+t_post_local
# Preallocate storage for the evolution
Rho_SE_post = np.zeros((num_isochromats,4,num_post_frames),dtype=complex)

# Evolve over 2nd after pulsetime
for j in tqdm(range(num_isochromats),desc="Evolving isochromats after 2nd pulse"):
    for i in range(num_post_frames):
        propagator = expm(Bloch_Operator_iso[j]*t_post_local[i])
        Rho_SE_post[j,:,i] = propagator@rho_SE_vec[j]

# Find the frame corresponding to the third-pulse time
pulse3_index = np.argmin(np.abs(t_post-time_for_pulse3))
actual_pulse3_time = t_post[pulse3_index]

# Pull every isochromat's state immediately before pulse 3
rho_SE_pre3_vec = Rho_SE_post[:, :, pulse3_index].copy()
# Shared evolution through the state immediately before pulse 3
Rho_shared_pre3 = Rho_SE_post[:, :, :pulse3_index+1]
# Apply the third 90-degree pulse
rho_SE_post3_vec = np.zeros((num_isochromats, 4), dtype=complex)
for j in range(num_isochromats):
    rho_SE_pre3_j = rho_SE_pre3_vec[j].reshape((2, 2), order="F")
    rho_SE_post3_j = Ux@rho_SE_pre3_j@Ux.conj().T
    rho_SE_post3_vec[j] = rho_SE_post3_j.flatten(order="F")
    
# Time-matched unpulsed branch beginning at the third-pulse time
Rho_SE_no3 = Rho_SE_post[:, :, pulse3_index:].copy()

# Reuse the remaining physical times as a local time beginning at pulse 3
t_post3 = t_post[pulse3_index:]
t_post3_local = t_post3-actual_pulse3_time
num_post3_frames = len(t_post3_local)

# Evolve from the state immediately after pulse 3
Rho_SE_post3 = np.zeros((num_isochromats, 4, num_post3_frames), dtype=complex)
for j in tqdm(range(num_isochromats), desc="Evolving isochromats after 3rd pulse"):
    for i in range(num_post3_frames):
        propagator = expm(Bloch_Operator_iso[j]*t_post3_local[i])
        Rho_SE_post3[j, :, i] = propagator@rho_SE_post3_vec[j]


# Join pulse histories along the time dimension
Rho_no_stimecho = np.concatenate((Pre_Rho_T,Rho_shared_pre3,Rho_SE_no3),axis=2)
Rho_stimecho = np.concatenate((Pre_Rho_T,Rho_shared_pre3,Rho_SE_post3),axis=2)

# Physical time excludes artificial introductory holds but includes pre/post pulse frames
physical_total_time = np.concatenate((physt, t_post[:pulse3_index+1], t_post3))

# Both trajectories have identical frame counts
num_frames = Rho_no_stimecho.shape[2]
assert Rho_stimecho.shape[2] == num_frames

# Preallocate expectation values for every isochromat
no3_ExpSx_iso = np.zeros((num_isochromats, num_frames))
no3_ExpSy_iso = np.zeros((num_isochromats, num_frames))
no3_ExpSz_iso = np.zeros((num_isochromats, num_frames))

stim_ExpSx_iso = np.zeros((num_isochromats, num_frames))
stim_ExpSy_iso = np.zeros((num_isochromats, num_frames))
stim_ExpSz_iso = np.zeros((num_isochromats, num_frames))

# Calculate expectation values separately for every isochromat
for j in tqdm(range(num_isochromats), desc="Computing isochromat expectation values"):
    for k in range(num_frames):
        rho_no3_jk = Rho_no_stimecho[j, :, k].reshape((2, 2), order="F")
        rho_stim_jk = Rho_stimecho[j, :, k].reshape((2, 2), order="F")

        no3_ExpSx_iso[j, k] = np.real(np.trace(rho_no3_jk@Sx))
        no3_ExpSy_iso[j, k] = np.real(np.trace(rho_no3_jk@Sy))
        no3_ExpSz_iso[j, k] = np.real(np.trace(rho_no3_jk@Sz))

        stim_ExpSx_iso[j, k] = np.real(np.trace(rho_stim_jk@Sx))
        stim_ExpSy_iso[j, k] = np.real(np.trace(rho_stim_jk@Sy))
        stim_ExpSz_iso[j, k] = np.real(np.trace(rho_stim_jk@Sz))

# Average vector components across the equally weighted isochromats
no3_ExpSx_t = np.mean(no3_ExpSx_iso, axis=0)
no3_ExpSy_t = np.mean(no3_ExpSy_iso, axis=0)
no3_ExpSz_t = np.mean(no3_ExpSz_iso, axis=0)

stim_ExpSx_t = np.mean(stim_ExpSx_iso, axis=0)
stim_ExpSy_t = np.mean(stim_ExpSy_iso, axis=0)
stim_ExpSz_t = np.mean(stim_ExpSz_iso, axis=0)

# Ensemble-averaged Bloch vectors
no3_rx_t = (2/hbar)*no3_ExpSx_t
no3_ry_t = (2/hbar)*no3_ExpSy_t
no3_rz_t = (2/hbar)*no3_ExpSz_t

stim_rx_t = (2/hbar)*stim_ExpSx_t
stim_ry_t = (2/hbar)*stim_ExpSy_t
stim_rz_t = (2/hbar)*stim_ExpSz_t

no3_R = np.column_stack((no3_rx_t, no3_ry_t, no3_rz_t))
stim_R = np.column_stack((stim_rx_t, stim_ry_t, stim_rz_t))

# Convert ensemble-averaged expectations into voxel magnetization
V_voxel = dx*dy*dz
N_voxel = nH_tissue*V_voxel

no3_Mx_t = N_voxel*H1gamma*no3_ExpSx_t
no3_My_t = N_voxel*H1gamma*no3_ExpSy_t
no3_Mz_t = N_voxel*H1gamma*no3_ExpSz_t

stim_Mx_t = N_voxel*H1gamma*stim_ExpSx_t
stim_My_t = N_voxel*H1gamma*stim_ExpSy_t
stim_Mz_t = N_voxel*H1gamma*stim_ExpSz_t

no3_M = np.column_stack((no3_Mx_t, no3_My_t, no3_Mz_t))
stim_M = np.column_stack((stim_Mx_t, stim_My_t, stim_Mz_t))

# Magnitude of the ensemble-averaged transverse vector
no3_Mxy = np.sqrt(no3_Mx_t**2+no3_My_t**2)
stim_Mxy = np.sqrt(stim_Mx_t**2+stim_My_t**2)

# --------------------------------------------------------- Plotting Extravaganza -------------------------------------------------------------- #

# Build complete animation-time vector, including introductory hold frames
animation_time = np.concatenate((np.zeros(repeatsum),physical_total_time))
num_total_frames = len(animation_time)

# Important raw-frame indices
pulse2_pre_idx = repeatsum+num_pre_frames-1
pulse2_post_idx = pulse2_pre_idx+1
pulse3_pre_idx = repeatsum+num_pre_frames+pulse3_index
pulse3_post_idx = pulse3_pre_idx+1
no3_echo_time = 2.0*time_for_pulse2
no3_echo_idx = np.argmin(np.abs(animation_time-no3_echo_time))
stim_echo_time = actual_pulse3_time+time_for_pulse2
stim_echo_idx = pulse3_post_idx+np.argmin(np.abs(t_post3-stim_echo_time))

# Thin physical frames while retaining holds, pulse states, stimulated echo, and final state
plot_stride = 5
plot_indices = np.concatenate((np.arange(repeatsum),np.arange(repeatsum,num_total_frames,plot_stride),[pulse2_pre_idx,pulse2_post_idx,pulse3_pre_idx,pulse3_post_idx,no3_echo_idx,stim_echo_idx,num_total_frames-1]))
plot_indices = np.unique(plot_indices)

# Thinned animation quantities
t_plot = animation_time[plot_indices]

no3_Mx_plot = no3_Mx_t[plot_indices]
no3_My_plot = no3_My_t[plot_indices]
no3_Mz_plot = no3_Mz_t[plot_indices]
no3_Mxy_plot = no3_Mxy[plot_indices]

stim_Mx_plot = stim_Mx_t[plot_indices]
stim_My_plot = stim_My_t[plot_indices]
stim_Mz_plot = stim_Mz_t[plot_indices]
stim_Mxy_plot = stim_Mxy[plot_indices]

# Effective phase-evolution times for the two selected coherence pathways
def no3_effective_time(t):
    return t if t <= time_for_pulse2 else 2.0*time_for_pulse2-t

def stim_effective_time(t):
    if t <= time_for_pulse2:
        return t
    if t <= actual_pulse3_time:
        return time_for_pulse2
    return time_for_pulse2-(t-actual_pulse3_time)

path_time = np.unique(np.concatenate((physical_total_time, [0.0, time_for_pulse2, actual_pulse3_time, stim_echo_time, simulation_end])))
path_time = path_time[(path_time >= 0.0) & (path_time <= simulation_end)]
# Nine representative isochromats show the actual phase spread without obscuring the pathways
display_iso_indices = np.linspace(0,num_isochromats-1,9,dtype=int)
display_delta_omega = delta_omega[display_iso_indices]
num_display_iso = len(display_iso_indices)
rainbow_colors = [f"hsl({int(280*j/(num_display_iso-1))},85%,60%)" for j in range(num_display_iso)]
no3_effective_path_time = np.array([no3_effective_time(t) for t in path_time])
stim_effective_path_time = np.array([stim_effective_time(t) for t in path_time])
no3_iso_path = display_delta_omega[:,None]*no3_effective_path_time[None,:]
stim_iso_path = display_delta_omega[:,None]*stim_effective_path_time[None,:]

def no3_iso_positions(t):
    return display_delta_omega*no3_effective_time(t)

def stim_iso_positions(t):
    return display_delta_omega*stim_effective_time(t)

# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
no3_red = "#FF4D4D"
stim_green = "#39FF14"
no3_circle = "#FFAAAA"
stim_circle = "#B6FF9C"
current_magenta = "#FF40FF"
grid_white = "rgba(255,255,255,0.18)"
circle_white = "#FFFFFF"
trail_cyan = "#4DDFFF"
circle_stride = 8

# Animation settings
frame_duration_ms = 55
theta_circle = np.linspace(0.0,2.0*np.pi,120)
num_plot_frames = len(plot_indices)

# Shared three-dimensional axis limit
all_magnetization = np.concatenate((no3_M.ravel(),stim_M.ravel()))
axis_limit = 1.10*np.max(np.abs(all_magnetization))
if axis_limit == 0:
    axis_limit = 1.0

# Shared pathway-axis limits
path_y_min = 1.10*min(np.min(no3_iso_path),np.min(stim_iso_path),0.0)
path_y_max = 1.15*max(np.max(no3_iso_path),np.max(stim_iso_path),1.0)

# Construct transverse circle at the current longitudinal position
def transverse_circle(mx,my,mz):
    radius = np.sqrt(mx**2+my**2)
    return radius*np.cos(theta_circle),radius*np.sin(theta_circle),np.full(theta_circle.shape,mz)

# Construct title for each animation frame
def get_frame_title(k):
    raw_idx = plot_indices[k]
    current_time = t_plot[k]

    if raw_idx < frame1repeats:
        return "Thermal-equilibrium magnetization"

    if raw_idx < repeatsum:
        return "Immediately after 90° excitation"

    if raw_idx == pulse2_pre_idx:
        return f"Immediately before second 90° pulse → t = {time_for_pulse2:.4f} s"

    if raw_idx == pulse2_post_idx:
        return f"Immediately after second 90° pulse → t = {time_for_pulse2:.4f} s"

    if raw_idx == pulse3_pre_idx:
        return f"Immediately before third 90° pulse → t = {actual_pulse3_time:.4f} s"

    if raw_idx == pulse3_post_idx:
        return f"Third 90° applied only to stimulated-echo branch → t = {actual_pulse3_time:.4f} s"

    if raw_idx == no3_echo_idx:
        return f"Selected two-pulse pathway rephases → t = {no3_echo_time:.4f} s"

    if raw_idx == stim_echo_idx:
        return f"Stimulated echo → t = {stim_echo_time:.4f} s"

    return f"Two-pulse and stimulated-echo evolution → t = {current_time:.4f} s"

# Create two paired rows: each spinor panel sits beside its selected pathway
fig = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "scene"},{"type": "xy"}],[{"type": "scene"},{"type": "xy"}]],
    column_widths=[0.46,0.54],
    horizontal_spacing=0.07,
    vertical_spacing=0.10,
    subplot_titles=("Two-Pulse / No-Recall Precession","Two-Pulse Selected Pathway","Stimulated-Echo Precession","Stimulated-Echo Selected Pathway")
)

# Initial transverse circles
no3_circle_initial = transverse_circle(no3_Mx_plot[0],no3_My_plot[0],no3_Mz_plot[0])
stim_circle_initial = transverse_circle(stim_Mx_plot[0],stim_My_plot[0],stim_Mz_plot[0])

# Trace 0: no3 magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0,no3_Mx_plot[0]],y=[0.0,no3_My_plot[0]],z=[0.0,no3_Mz_plot[0]],mode="lines+markers",line=dict(color=no3_red,width=8),marker=dict(size=[2,6],color=no3_red),name="No-recall magnetization",showlegend=False,hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"),row=1,col=1)
# Trace 1: no3 current transverse circle
fig.add_trace(go.Scatter3d(x=no3_circle_initial[0],y=no3_circle_initial[1],z=no3_circle_initial[2],mode="lines",line=dict(color=no3_red,width=4),showlegend=False,hoverinfo="skip"),row=1,col=1)
# Trace 2: Stimulated-echo magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0,stim_Mx_plot[0]],y=[0.0,stim_My_plot[0]],z=[0.0,stim_Mz_plot[0]],mode="lines+markers",line=dict(color=stim_green,width=8),marker=dict(size=[2,6],color=stim_green),name="Stimulated magnetization",showlegend=False,hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"),row=2,col=1)
# Trace 3: Stimulated current transverse circle
fig.add_trace(go.Scatter3d(x=stim_circle_initial[0],y=stim_circle_initial[1],z=stim_circle_initial[2],mode="lines",line=dict(color=stim_green,width=4),showlegend=False,hoverinfo="skip"),row=2,col=1)
# Traces 4-5: branch-style legend keys; the actual paths are the rainbow strands below
# Trace 4: no-third-pulse branch legend key
fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=no3_red,size=8,symbol="circle"),name="No third pulse"),row=1,col=2)
# Trace 5: stimulated-echo branch legend key
fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=stim_green,size=8,symbol="diamond"),name="Stimulated echo"),row=2,col=2)
# Trace 6: second-pulse reference in the no-third-pulse panel
fig.add_trace(go.Scatter(x=[time_for_pulse2,time_for_pulse2],y=[path_y_min,path_y_max],mode="lines",line=dict(color=text_white,width=2),name="Second 90° pulse",hoverinfo="skip"),row=1,col=2)
# Trace 7: second-pulse reference in the stimulated-echo panel
fig.add_trace(go.Scatter(x=[time_for_pulse2,time_for_pulse2],y=[path_y_min,path_y_max],mode="lines",line=dict(color=text_white,width=2),name="Second 90° pulse",hoverinfo="skip",showlegend=False),row=2,col=2)
# Trace 8: third-pulse channel-switch reference
fig.add_trace(go.Scatter(x=[actual_pulse3_time,actual_pulse3_time],y=[path_y_min,path_y_max],mode="lines",line=dict(color=current_magenta,width=2),name="Third 90° pulse",hoverinfo="skip"),row=2,col=2)
# Traces 9-10: zero-phase/rephasing references
fig.add_trace(go.Scatter(x=[0.0,simulation_end],y=[0.0,0.0],mode="lines",line=dict(color=axis_gold,width=2,dash="dash"),name="Zero phase / echo",hoverinfo="skip"),row=1,col=2)
fig.add_trace(go.Scatter(x=[0.0,simulation_end],y=[0.0,0.0],mode="lines",line=dict(color=axis_gold,width=2,dash="dash"),name="Zero phase / echo",hoverinfo="skip",showlegend=False),row=2,col=2)
# Traces 11-12: moving pathway points
fig.add_trace(go.Scatter(x=np.full(num_display_iso,t_plot[0]),y=no3_iso_positions(t_plot[0]),mode="markers",marker=dict(color=rainbow_colors,size=7,line=dict(color=no3_red,width=2)),name="Current no-pulse isochromats",showlegend=False,hoverinfo="skip"),row=1,col=2)
fig.add_trace(go.Scatter(x=np.full(num_display_iso,t_plot[0]),y=stim_iso_positions(t_plot[0]),mode="markers",marker=dict(color=rainbow_colors,size=7,symbol="diamond",line=dict(color=stim_green,width=2)),name="Current stimulated isochromats",showlegend=False,hoverinfo="skip"),row=2,col=2)
# Trace 13: no3 transverse-circle history
fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="lines",line=dict(color=no3_red,width=2),showlegend=False,hoverinfo="skip"),row=1,col=1)
# Trace 14: Stimulated transverse-circle history
fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="lines",line=dict(color=stim_green,width=2),showlegend=False,hoverinfo="skip"),row=2,col=1)

# Rainbow isochromat strands remain close to each bold selected pathway
for j in range(num_display_iso):
    offset_hz_j = display_delta_omega[j]/(2*np.pi)
    hover = f"Δf = {offset_hz_j:.2f} Hz<extra></extra>"
    fig.add_trace(go.Scatter(x=path_time,y=no3_iso_path[j],mode="lines",line=dict(color=rainbow_colors[j],width=2),opacity=0.75,showlegend=False,hovertemplate=hover),row=1,col=2)
    fig.add_trace(go.Scatter(x=path_time,y=stim_iso_path[j],mode="lines",line=dict(color=rainbow_colors[j],width=2,dash="dot"),opacity=0.75,showlegend=False,hovertemplate=hover),row=2,col=2)

# Separate accumulated transverse-circle histories
no3_trail_x = []
no3_trail_y = []
no3_trail_z = []

stim_trail_x = []
stim_trail_y = []
stim_trail_z = []

animation_frames = []

for k in tqdm(range(num_plot_frames),desc="Building Stimulated-echo animation",unit="frame",dynamic_ncols=True):
    no3_circle = transverse_circle(no3_Mx_plot[k],no3_My_plot[k],no3_Mz_plot[k])
    stim_circle = transverse_circle(stim_Mx_plot[k],stim_My_plot[k],stim_Mz_plot[k])

    # Accumulate historical circles only during physical evolution
    raw_idx = plot_indices[k]
    if raw_idx >= repeatsum:
        relative_frame = raw_idx-repeatsum

        if relative_frame % circle_stride == 0:
            no3_trail_x.extend(no3_circle[0].tolist())
            no3_trail_x.append(None)
            no3_trail_y.extend(no3_circle[1].tolist())
            no3_trail_y.append(None)
            no3_trail_z.extend(no3_circle[2].tolist())
            no3_trail_z.append(None)

            stim_trail_x.extend(stim_circle[0].tolist())
            stim_trail_x.append(None)
            stim_trail_y.extend(stim_circle[1].tolist())
            stim_trail_y.append(None)
            stim_trail_z.extend(stim_circle[2].tolist())
            stim_trail_z.append(None)

    no3_frame_trail_x = no3_trail_x.copy() if no3_trail_x else [None]
    no3_frame_trail_y = no3_trail_y.copy() if no3_trail_y else [None]
    no3_frame_trail_z = no3_trail_z.copy() if no3_trail_z else [None]

    stim_frame_trail_x = stim_trail_x.copy() if stim_trail_x else [None]
    stim_frame_trail_y = stim_trail_y.copy() if stim_trail_y else [None]
    stim_frame_trail_z = stim_trail_z.copy() if stim_trail_z else [None]

    animation_frames.append(go.Frame(
        name=str(k),
        data=[
            go.Scatter3d(x=[0.0,no3_Mx_plot[k]],y=[0.0,no3_My_plot[k]],z=[0.0,no3_Mz_plot[k]]),
            go.Scatter3d(x=no3_circle[0],y=no3_circle[1],z=no3_circle[2]),
            go.Scatter3d(x=[0.0,stim_Mx_plot[k]],y=[0.0,stim_My_plot[k]],z=[0.0,stim_Mz_plot[k]]),
            go.Scatter3d(x=stim_circle[0],y=stim_circle[1],z=stim_circle[2]),
            go.Scatter(x=np.full(num_display_iso,t_plot[k]),y=no3_iso_positions(t_plot[k])),
            go.Scatter(x=np.full(num_display_iso,t_plot[k]),y=stim_iso_positions(t_plot[k])),
            go.Scatter3d(x=no3_frame_trail_x,y=no3_frame_trail_y,z=no3_frame_trail_z),
            go.Scatter3d(x=stim_frame_trail_x,y=stim_frame_trail_y,z=stim_frame_trail_z)
        ],
        traces=[0,1,2,3,11,12,13,14],
        layout=go.Layout(title=dict(text=get_frame_title(k)))
    ))

fig.frames = animation_frames

# Slider controls
slider_steps = []

for k in range(num_plot_frames):
    slider_steps.append(dict(method="animate",args=[[str(k)],{"mode": "immediate","frame": {"duration": 0,"redraw": True},"transition": {"duration": 0}}],label=""))

# Shared Bloch-scene styling
scene_style = dict(
    bgcolor=panel_background,
    aspectmode="cube",
    dragmode="orbit",
    uirevision="constant",
    camera=dict(eye=dict(x=1.6,y=1.6,z=0.55),up=dict(x=0,y=0,z=1)),
    xaxis=dict(title="Mₓ",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True),
    yaxis=dict(title="Mᵧ",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True),
    zaxis=dict(title="M_z",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True)
)

# Complete layout
fig.update_layout(
    title=dict(text=get_frame_title(0),x=0.5,xanchor="center",font=dict(size=22,color=text_white)),
	height=860,
	margin=dict(l=25,r=25,t=90,b=105),
	#height=950,
	#margin=dict(l=30,r=30,t=110,b=135),
    paper_bgcolor=page_background,
    plot_bgcolor=panel_background,
    font=dict(color=text_white),
    scene=scene_style,
    scene2=scene_style,
    xaxis=dict(title="Time (s)",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,range=[0.0,simulation_end]),
    yaxis=dict(title="Isochromat phase (rad)",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,range=[path_y_min,path_y_max]),
    xaxis2=dict(title="Time (s)",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,range=[0.0,simulation_end]),
    yaxis2=dict(title="Isochromat phase (rad)",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,range=[path_y_min,path_y_max]),
    legend=dict(x=0.59,y=1,xanchor="left",yanchor="top",orientation="h",bgcolor="rgba(6,24,43,0.70)",bordercolor=axis_gold,borderwidth=1,font=dict(color=text_white,size=8),itemsizing="constant",itemwidth=30,tracegroupgap=0),
    updatemenus=[dict(type="buttons",direction="left",showactive=False,x=0.42,y=-0.15,xanchor="center",yanchor="top",bgcolor=panel_background,bordercolor=axis_gold,font=dict(color=text_white),buttons=[
        dict(label="▶ Play",method="animate",args=[None,{"fromcurrent": True,"mode": "immediate","frame": {"duration": frame_duration_ms,"redraw": True},"transition": {"duration": 0}}]),
        dict(label="❚❚ Pause",method="animate",args=[[None],{"mode": "immediate","frame": {"duration": 0,"redraw": False},"transition": {"duration": 0}}])
    ])],
    sliders=[dict(active=0,x=0.10,len=0.82,y=-0.035,xanchor="left",yanchor="top",bgcolor=panel_background,bordercolor=axis_gold,activebgcolor=current_magenta,tickcolor=axis_gold,font=dict(color=text_white),currentvalue=dict(visible=False),pad=dict(t=30,b=0),steps=slider_steps)]
)

fig.update_annotations(font=dict(color=axis_gold,size=17))

# Labels for the selected coherence-channel behaviors
fig.add_annotation(x=0.45*time_for_pulse2,y=0.65*path_y_max,text="Transverse dephasing",showarrow=False,font=dict(color=text_white,size=11),xref="x",yref="y")
fig.add_annotation(x=0.5*(time_for_pulse2+actual_pulse3_time),y=0.45*path_y_max,text="Rephasing",showarrow=False,font=dict(color=text_white,size=11),xref="x",yref="y")
fig.add_annotation(x=no3_echo_time,y=0.0,text="Two-pulse echo",showarrow=True,arrowhead=2,ax=-48,ay=42,arrowcolor=axis_gold,font=dict(color=text_white,size=11),xref="x",yref="y")
fig.add_annotation(x=0.45*time_for_pulse2,y=0.65*path_y_max,text="Transverse dephasing",showarrow=False,font=dict(color=text_white,size=11),xref="x2",yref="y2")
fig.add_annotation(x=0.5*(time_for_pulse2+actual_pulse3_time),y=0.65*path_y_max,text="Z storage",showarrow=False,font=dict(color=text_white,size=11),xref="x2",yref="y2")
fig.add_annotation(x=0.5*(actual_pulse3_time+stim_echo_time),y=0.45*path_y_max,text="Rephasing",showarrow=False,font=dict(color=text_white,size=11),xref="x2",yref="y2")
fig.add_annotation(x=stim_echo_time,y=0.0,text="Stimulated echo",showarrow=True,arrowhead=2,ax=50,ay=-38,arrowcolor=axis_gold,font=dict(color=text_white,size=11),xref="x2",yref="y2")
fig.add_annotation(x=0.9*simulation_end,y=-0.85*path_y_max,text="9 representative off-resonance isochromats",showarrow=False,xanchor="right",font=dict(color=axis_gold,size=9),xref="x",yref="y")
fig.add_annotation(x=0.9*simulation_end,y=-0.85*path_y_max,text="9 representative off-resonance isochromats",showarrow=False,xanchor="right",font=dict(color=axis_gold,size=9),xref="x2",yref="y2")


# Gold borders around both Bloch panels
for scene_name in ("scene","scene2"):
    scene_domain = getattr(fig.layout,scene_name).domain
    fig.add_shape(type="rect",xref="paper",yref="paper",x0=scene_domain.x[0],x1=scene_domain.x[1],y0=scene_domain.y[0],y1=scene_domain.y[1],line=dict(color=axis_gold,width=1),fillcolor="rgba(0,0,0,0)",layer="above")

print("Writing interactive spinor and coherence-pathway HTML file...")

fig.write_html(
    "06_stimulated_echo_coherence_pathways.html",
    auto_open=True,
    auto_play=False,
    config={"scrollZoom": True,"displaylogo": False,"responsive": True},
    post_script="document.title = '06_stimulated_echo_coherence_pathways';"
)

print("HTML file complete.")
