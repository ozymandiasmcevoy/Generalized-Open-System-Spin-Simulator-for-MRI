# This script constructs a tissue-configurable, quasi-classical simulator of voxel-scale Hahn spin-echo dynamics. (Ozymandias McEvoy)
# The excitation and refocusing pulses are modeled as ideal, instantaneous 90-degree and 180-degree unitary rotations.
# T1 and T2 relaxation are modeled through Lindblad channels, while static off-resonance variation is represented by an ensemble of isochromats.
# The simulation compares uninterrupted free dephasing with Hahn-echo refocusing over two echo times.


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

# Define Echo Pulse via Unitary Quantum Rotation
SE = expm((-1j*np.pi*Sx)/hbar)


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
time_for_hahn = TE/2
simulation_end = 2*TE

# Establish the physical sampling interval from the pre-pulse segment
num_pre_frames = 236
physt = np.linspace(0.0,time_for_hahn,num_pre_frames)
sample_dt = physt[1]-physt[0]
# Automatically preserve that sampling interval after the 180-degree pulse
post_duration = simulation_end-time_for_hahn
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
for j in tqdm(range(num_isochromats),desc="Evolving isochromats before refocusing"):
    for i in range(num_pre_frames):
        Pre_Rho_T[j,:,i+repeatsum] = expm(Bloch_Operator_iso[j]*physt[i])@rho_plus_vec

# Obtain every isochromat's state immediately before the 180-degree pulse
rho_tau_vec = Pre_Rho_T[:,:,-1]
# Create Copy for further evolution without echo
rho_control_vec = rho_tau_vec.copy()
# Pulse the midpoint state and reflatten all
rho_SE_vec = np.zeros((num_isochromats,4),dtype=complex)
for j in range(num_isochromats):
    rho_tau_j = rho_tau_vec[j].reshape((2,2),order="F")
    rho_SE_j = SE@rho_tau_j@SE.conj().T
    rho_SE_vec[j] = rho_SE_j.flatten(order="F")

# Establish Length of new time evolution
t_post_local = np.linspace(0.0,simulation_end-time_for_hahn,num_post_frames)
t_post = time_for_hahn+t_post_local
# Preallocate storage for the two paths
Rho_SE_post = np.zeros((num_isochromats,4,num_post_frames),dtype=complex)
Rho_control_post = np.zeros((num_isochromats,4,num_post_frames),dtype=complex)
# Evolve over both paths
for j in tqdm(range(num_isochromats),desc="Evolving isochromats after refocusing"):
    for i in range(num_post_frames):
        propagator = expm(Bloch_Operator_iso[j]*t_post_local[i])
        Rho_SE_post[j,:,i] = propagator@rho_SE_vec[j]
        Rho_control_post[j,:,i] = propagator@rho_control_vec[j]

# Join pre- and post-refocusing histories along the time dimension
Rho_SE_T = np.concatenate((Pre_Rho_T,Rho_SE_post),axis=2)
Rho_control_T = np.concatenate((Pre_Rho_T,Rho_control_post),axis=2)

# Physical time excludes artificial introductory holds
physical_total_time = np.concatenate((physt,t_post))

# Number of stored frames
num_frames = Rho_control_T.shape[2]

# Preallocate expectation values for every isochromat
cntrl_ExpSx_iso = np.zeros((num_isochromats,num_frames))
cntrl_ExpSy_iso = np.zeros((num_isochromats,num_frames))
cntrl_ExpSz_iso = np.zeros((num_isochromats,num_frames))

hahn_ExpSx_iso = np.zeros((num_isochromats,num_frames))
hahn_ExpSy_iso = np.zeros((num_isochromats,num_frames))
hahn_ExpSz_iso = np.zeros((num_isochromats,num_frames))

# Calculate expectation values separately for every isochromat
for j in tqdm(range(num_isochromats),desc="Computing isochromat expectation values"):
    for k in range(num_frames):
        rho_control_jk = Rho_control_T[j,:,k].reshape((2,2),order="F")
        rho_hahn_jk = Rho_SE_T[j,:,k].reshape((2,2),order="F")

        cntrl_ExpSx_iso[j,k] = np.real(np.trace(rho_control_jk@Sx))
        cntrl_ExpSy_iso[j,k] = np.real(np.trace(rho_control_jk@Sy))
        cntrl_ExpSz_iso[j,k] = np.real(np.trace(rho_control_jk@Sz))

        hahn_ExpSx_iso[j,k] = np.real(np.trace(rho_hahn_jk@Sx))
        hahn_ExpSy_iso[j,k] = np.real(np.trace(rho_hahn_jk@Sy))
        hahn_ExpSz_iso[j,k] = np.real(np.trace(rho_hahn_jk@Sz))

# Average vector components across the equally weighted isochromats
cntrl_ExpSx_t = np.mean(cntrl_ExpSx_iso,axis=0)
cntrl_ExpSy_t = np.mean(cntrl_ExpSy_iso,axis=0)
cntrl_ExpSz_t = np.mean(cntrl_ExpSz_iso,axis=0)

hahn_ExpSx_t = np.mean(hahn_ExpSx_iso,axis=0)
hahn_ExpSy_t = np.mean(hahn_ExpSy_iso,axis=0)
hahn_ExpSz_t = np.mean(hahn_ExpSz_iso,axis=0)

# Ensemble-averaged Bloch vectors
cntrl_rx_t = (2/hbar)*cntrl_ExpSx_t
cntrl_ry_t = (2/hbar)*cntrl_ExpSy_t
cntrl_rz_t = (2/hbar)*cntrl_ExpSz_t

hahn_rx_t = (2/hbar)*hahn_ExpSx_t
hahn_ry_t = (2/hbar)*hahn_ExpSy_t
hahn_rz_t = (2/hbar)*hahn_ExpSz_t

cntrl_R = np.column_stack((cntrl_rx_t,cntrl_ry_t,cntrl_rz_t))
hahn_R = np.column_stack((hahn_rx_t,hahn_ry_t,hahn_rz_t))

# Convert ensemble-averaged expectations into voxel magnetization
V_voxel = dx*dy*dz
N_voxel = nH_tissue*V_voxel

cntrl_Mx_t = N_voxel*H1gamma*cntrl_ExpSx_t
cntrl_My_t = N_voxel*H1gamma*cntrl_ExpSy_t
cntrl_Mz_t = N_voxel*H1gamma*cntrl_ExpSz_t

hahn_Mx_t = N_voxel*H1gamma*hahn_ExpSx_t
hahn_My_t = N_voxel*H1gamma*hahn_ExpSy_t
hahn_Mz_t = N_voxel*H1gamma*hahn_ExpSz_t

cntrl_M = np.column_stack((cntrl_Mx_t,cntrl_My_t,cntrl_Mz_t))
hahn_M = np.column_stack((hahn_Mx_t,hahn_My_t,hahn_Mz_t))

# Magnitude of the ensemble-averaged transverse vector
cntrl_Mxy = np.sqrt(cntrl_Mx_t**2+cntrl_My_t**2)
hahn_Mxy = np.sqrt(hahn_Mx_t**2+hahn_My_t**2)



# --------------------------------------------------------- Plotting Extravaganza -------------------------------------------------------------- #

# Build complete animation-time vector, including introductory hold frames
animation_time = np.concatenate((np.zeros(repeatsum),physical_total_time))
num_total_frames = len(animation_time)

# Important raw-frame indices
pre_pulse_end_idx = repeatsum+num_pre_frames-1
post_pulse_start_idx = pre_pulse_end_idx+1
echo_idx = repeatsum+num_pre_frames+np.argmin(np.abs(t_post-TE))

# Thin physical frames while retaining holds, pulse states, echo, and final state
plot_stride = 5
plot_indices = np.concatenate((np.arange(repeatsum),np.arange(repeatsum,num_total_frames,plot_stride),[pre_pulse_end_idx,post_pulse_start_idx,echo_idx,num_total_frames-1]))
plot_indices = np.unique(plot_indices)

# Thinned animation quantities
t_plot = animation_time[plot_indices]

cntrl_Mx_plot = cntrl_Mx_t[plot_indices]
cntrl_My_plot = cntrl_My_t[plot_indices]
cntrl_Mz_plot = cntrl_Mz_t[plot_indices]
cntrl_Mxy_plot = cntrl_Mxy[plot_indices]

hahn_Mx_plot = hahn_Mx_t[plot_indices]
hahn_My_plot = hahn_My_t[plot_indices]
hahn_Mz_plot = hahn_Mz_t[plot_indices]
hahn_Mxy_plot = hahn_Mxy[plot_indices]

# Physical signal curves exclude introductory artificial holds
cntrl_Mxy_phys = cntrl_Mxy[repeatsum:]
hahn_Mxy_phys = hahn_Mxy[repeatsum:]

# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
control_red = "#FF4D4D"
hahn_green = "#39FF14"
control_circle = "#FFAAAA"
hahn_circle = "#B6FF9C"
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
all_magnetization = np.concatenate((cntrl_M.ravel(),hahn_M.ravel()))
axis_limit = 1.10*np.max(np.abs(all_magnetization))
if axis_limit == 0:
    axis_limit = 1.0

# Shared signal-axis limit
signal_y_max = 1.10*max(np.max(cntrl_Mxy),np.max(hahn_Mxy))
if signal_y_max == 0:
    signal_y_max = 1.0

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

    if raw_idx == pre_pulse_end_idx:
        return f"Immediately before 180° pulse → t = TE/2 = {time_for_hahn:.4f} s"

    if raw_idx == post_pulse_start_idx:
        return f"Immediately after 180° pulse → t = TE/2 = {time_for_hahn:.4f} s"

    if raw_idx == echo_idx:
        return f"Echo time → TE = {TE:.4f} s"

    return f"Control and Hahn-echo evolution → t = {current_time:.4f} s"

# Create two stacked Bloch panels and one shared signal panel
fig = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "scene"},{"type": "xy","rowspan": 2}],[{"type": "scene"},None]],
    column_widths=[0.48,0.52],
    horizontal_spacing=0.07,
    vertical_spacing=0.10,
    subplot_titles=("Control Precession","Transverse Signal Comparison","Hahn-Echo Precession")
)

# Initial transverse circles
cntrl_circle_initial = transverse_circle(cntrl_Mx_plot[0],cntrl_My_plot[0],cntrl_Mz_plot[0])
hahn_circle_initial = transverse_circle(hahn_Mx_plot[0],hahn_My_plot[0],hahn_Mz_plot[0])

# Trace 0: control magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0,cntrl_Mx_plot[0]],y=[0.0,cntrl_My_plot[0]],z=[0.0,cntrl_Mz_plot[0]],mode="lines+markers",line=dict(color=control_red,width=8),marker=dict(size=[2,6],color=control_red),name="Control magnetization",showlegend=False,hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"),row=1,col=1)
# Trace 1: control current transverse circle
fig.add_trace(go.Scatter3d(x=cntrl_circle_initial[0],y=cntrl_circle_initial[1],z=cntrl_circle_initial[2],mode="lines",line=dict(color=circle_white,width=6),showlegend=False,hoverinfo="skip"),row=1,col=1)
# Trace 2: Hahn-echo magnetization vector
fig.add_trace(go.Scatter3d(x=[0.0,hahn_Mx_plot[0]],y=[0.0,hahn_My_plot[0]],z=[0.0,hahn_Mz_plot[0]],mode="lines+markers",line=dict(color=hahn_green,width=8),marker=dict(size=[2,6],color=hahn_green),name="Hahn magnetization",showlegend=False,hovertemplate="Mx = %{x:.3e}<br>My = %{y:.3e}<br>Mz = %{z:.3e}<extra></extra>"),row=2,col=1)
# Trace 3: Hahn current transverse circle
fig.add_trace(go.Scatter3d(x=hahn_circle_initial[0],y=hahn_circle_initial[1],z=hahn_circle_initial[2],mode="lines",line=dict(color=circle_white,width=6),showlegend=False,hoverinfo="skip"),row=2,col=1)
# Trace 4: complete control signal
fig.add_trace(go.Scatter(x=physical_total_time,y=cntrl_Mxy_phys,mode="lines",line=dict(color=control_red,width=3),name="Control M⊥",hovertemplate="Time = %{x:.4f} s<br>Control M⊥ = %{y:.3e}<extra></extra>"),row=1,col=2)
# Trace 5: complete Hahn-echo signal
fig.add_trace(go.Scatter(x=physical_total_time,y=hahn_Mxy_phys,mode="lines",line=dict(color=hahn_green,width=3),name="Hahn echo M⊥",hovertemplate="Time = %{x:.4f} s<br>Hahn M⊥ = %{y:.3e}<extra></extra>"),row=1,col=2)
# Trace 6: refocusing-pulse reference
fig.add_trace(go.Scatter(x=[time_for_hahn,time_for_hahn],y=[0.0,signal_y_max],mode="lines",line=dict(color=text_white,width=2,dash="dot"),name="180° pulse at TE/2",hoverinfo="skip"),row=1,col=2)
# Trace 7: echo-time reference
fig.add_trace(go.Scatter(x=[TE,TE],y=[0.0,signal_y_max],mode="lines",line=dict(color=axis_gold,width=2,dash="dash"),name="Echo at TE",hoverinfo="skip"),row=1,col=2)
# Traces 8 and 9: moving signal points
fig.add_trace(go.Scatter(x=[t_plot[0]],y=[cntrl_Mxy_plot[0]],mode="markers",marker=dict(color=control_red,size=11,line=dict(color="white",width=1)),name="Current control signal",showlegend=False,hoverinfo="skip"),row=1,col=2)
fig.add_trace(go.Scatter(x=[t_plot[0]],y=[hahn_Mxy_plot[0]],mode="markers",marker=dict(color=hahn_green,size=11,line=dict(color="white",width=1)),name="Current Hahn signal",showlegend=False,hoverinfo="skip"),row=1,col=2)
# Trace 10: control transverse-circle history
fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="lines",line=dict(color=trail_cyan,width=2),showlegend=False,hoverinfo="skip"),row=1,col=1)
# Trace 11: Hahn transverse-circle history
fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="lines",line=dict(color=trail_cyan,width=2),showlegend=False,hoverinfo="skip"),row=2,col=1)

# Separate accumulated transverse-circle histories
cntrl_trail_x = []
cntrl_trail_y = []
cntrl_trail_z = []

hahn_trail_x = []
hahn_trail_y = []
hahn_trail_z = []

animation_frames = []

for k in tqdm(range(num_plot_frames),desc="Building Hahn-echo animation",unit="frame",dynamic_ncols=True):
    cntrl_circle = transverse_circle(cntrl_Mx_plot[k],cntrl_My_plot[k],cntrl_Mz_plot[k])
    hahn_circle = transverse_circle(hahn_Mx_plot[k],hahn_My_plot[k],hahn_Mz_plot[k])

    # Accumulate historical circles only during physical evolution
    if k >= repeatsum:
        relative_frame = k-repeatsum

        if relative_frame % circle_stride == 0:
            cntrl_trail_x.extend(cntrl_circle[0].tolist())
            cntrl_trail_x.append(None)
            cntrl_trail_y.extend(cntrl_circle[1].tolist())
            cntrl_trail_y.append(None)
            cntrl_trail_z.extend(cntrl_circle[2].tolist())
            cntrl_trail_z.append(None)

            hahn_trail_x.extend(hahn_circle[0].tolist())
            hahn_trail_x.append(None)
            hahn_trail_y.extend(hahn_circle[1].tolist())
            hahn_trail_y.append(None)
            hahn_trail_z.extend(hahn_circle[2].tolist())
            hahn_trail_z.append(None)

    cntrl_frame_trail_x = cntrl_trail_x.copy() if cntrl_trail_x else [None]
    cntrl_frame_trail_y = cntrl_trail_y.copy() if cntrl_trail_y else [None]
    cntrl_frame_trail_z = cntrl_trail_z.copy() if cntrl_trail_z else [None]

    hahn_frame_trail_x = hahn_trail_x.copy() if hahn_trail_x else [None]
    hahn_frame_trail_y = hahn_trail_y.copy() if hahn_trail_y else [None]
    hahn_frame_trail_z = hahn_trail_z.copy() if hahn_trail_z else [None]

    animation_frames.append(go.Frame(
        name=str(k),
        data=[
            go.Scatter3d(x=[0.0,cntrl_Mx_plot[k]],y=[0.0,cntrl_My_plot[k]],z=[0.0,cntrl_Mz_plot[k]]),
            go.Scatter3d(x=cntrl_circle[0],y=cntrl_circle[1],z=cntrl_circle[2]),
            go.Scatter3d(x=[0.0,hahn_Mx_plot[k]],y=[0.0,hahn_My_plot[k]],z=[0.0,hahn_Mz_plot[k]]),
            go.Scatter3d(x=hahn_circle[0],y=hahn_circle[1],z=hahn_circle[2]),
            go.Scatter(x=[t_plot[k]],y=[cntrl_Mxy_plot[k]]),
            go.Scatter(x=[t_plot[k]],y=[hahn_Mxy_plot[k]]),
            go.Scatter3d(x=cntrl_frame_trail_x,y=cntrl_frame_trail_y,z=cntrl_frame_trail_z),
            go.Scatter3d(x=hahn_frame_trail_x,y=hahn_frame_trail_y,z=hahn_frame_trail_z)
        ],
        traces=[0,1,2,3,8,9,10,11],
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
    camera=dict(eye=dict(x=1.45,y=-1.0,z=0.8)),
    xaxis=dict(title="Mₓ",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True),
    yaxis=dict(title="Mᵧ",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True),
    zaxis=dict(title="M_z",color=axis_gold,backgroundcolor=panel_background,gridcolor=grid_white,zerolinecolor=axis_gold,linecolor=axis_gold,range=[-axis_limit,axis_limit],tickformat=".1e",showbackground=True)
)

# Complete layout
fig.update_layout(
    title=dict(text=get_frame_title(0),x=0.5,xanchor="center",font=dict(size=22,color=text_white)),
	height=800,
	margin=dict(l=25,r=25,t=90,b=105),
	#height=950,
	#margin=dict(l=30,r=30,t=110,b=135),
    paper_bgcolor=page_background,
    plot_bgcolor=panel_background,
    font=dict(color=text_white),
    scene=scene_style,
    scene2=scene_style,
    xaxis=dict(title="Time (s)",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,range=[0.0,simulation_end]),
    yaxis=dict(title="Transverse magnetization, M⊥",color=axis_gold,linecolor=axis_gold,gridcolor=grid_white,zeroline=False,showline=True,mirror=True,tickformat=".2e",range=[-0.05*signal_y_max,signal_y_max]),
    legend=dict(x=0.99,y=0.98,xanchor="right",yanchor="top",bgcolor="rgba(6,24,43,0.88)",bordercolor=axis_gold,borderwidth=1,font=dict(color=text_white)),
    updatemenus=[dict(type="buttons",direction="left",showactive=False,x=0.42,y=-0.15,xanchor="center",yanchor="top",bgcolor=panel_background,bordercolor=axis_gold,font=dict(color=text_white),buttons=[
        dict(label="▶ Play",method="animate",args=[None,{"fromcurrent": True,"mode": "immediate","frame": {"duration": frame_duration_ms,"redraw": True},"transition": {"duration": 0}}]),
        dict(label="❚❚ Pause",method="animate",args=[[None],{"mode": "immediate","frame": {"duration": 0,"redraw": False},"transition": {"duration": 0}}])
    ])],
    sliders=[dict(active=0,x=0.10,len=0.82,y=-0.035,xanchor="left",yanchor="top",bgcolor=panel_background,bordercolor=axis_gold,activebgcolor=current_magenta,tickcolor=axis_gold,font=dict(color=text_white),currentvalue=dict(visible=False),pad=dict(t=30,b=0),steps=slider_steps)]
)

fig.update_annotations(font=dict(color=axis_gold,size=17))

# Gold borders around both Bloch panels
for scene_name in ("scene","scene2"):
    scene_domain = getattr(fig.layout,scene_name).domain
    fig.add_shape(type="rect",xref="paper",yref="paper",x0=scene_domain.x[0],x1=scene_domain.x[1],y0=scene_domain.y[0],y1=scene_domain.y[1],line=dict(color=axis_gold,width=1),fillcolor="rgba(0,0,0,0)",layer="above")

print("Writing interactive Hahn-echo HTML file...")

fig.write_html(
    "05_hahn_echo_rephasing.html",
    auto_open=True,
    auto_play=False,
    config={"scrollZoom": True,"displaylogo": False,"responsive": True},
    post_script="document.title = '05_hahn_echo_rephasing';"
)


print("HTML file complete.")