
# This script constructs a tissue-configurable, semiclassical, physics-based simulator of voxel-scale nuclear magnetic resonance Bloch dynamics. (Ozymandias McEvoy)
# The RF excitation is modeled as an ideal, instantaneous, single-flip-angle unitary rotation.
# Longitudinal and transverse relaxation are neglected so that the subsequent evolution is purely unitary under the Zeeman Hamiltonian.
# The script simulates voxel-scale magnetization precession in a static B0 field.

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

# -------------------------------------------------------------- Evolution of Post-Pulse State ------------------------------------------------------------------------ # 

# Declare introductory animation holds
frame1repeats = 10
frame2repeats = 15
repeatsum = frame1repeats + frame2repeats

# Simulate several laboratory-frame Larmor cycles
larmor_frequency_Hz = H1gamma * B0_norm / (2 * const.pi)
larmor_period = 1.0 / larmor_frequency_Hz
number_of_larmor_cycles = 50
num_physical_frames = 3600//2
physt = np.linspace(0.0, number_of_larmor_cycles * larmor_period, num_physical_frames, endpoint=False)

# Initialize density-matrix history, including artificial hold frames
Rho_T = np.zeros((2, 2, repeatsum + num_physical_frames), dtype=complex)

# Fill the first frames with the thermal-equilibrium state for plotting clarity
for i in range(frame1repeats):
    Rho_T[:, :, i] = rho_minus

# Fill the next frames with the immediately post-pulse state
for i in range(frame1repeats, repeatsum):
    Rho_T[:, :, i] = rho_plus

# Evolve and store the post-pulse state under the Zeeman Hamiltonian
for i, t in enumerate(tqdm(physt, desc="Evolving post-pulse state", unit="frame", dynamic_ncols=True)):
    U_t = expm((-1j * H * t) / hbar)
    Rho_T[:, :, i + repeatsum] = U_t @ rho_plus @ U_t.conj().T

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

# Expectation values during physical evolution only
ExpSx_phys = ExpSx_t[repeatsum:]
ExpSy_phys = ExpSy_t[repeatsum:]
ExpSz_phys = ExpSz_t[repeatsum:]

# Fourier transforms of those expectation-value time series
FT_Sx_t = np.fft.fft(ExpSx_phys)
FT_Sy_t = np.fft.fft(ExpSy_phys)
FT_Sz_t = np.fft.fft(ExpSz_phys)
FT_S = np.column_stack((FT_Sx_t, FT_Sy_t, FT_Sz_t))

# Corresponding Fourier frequencies
dt = physt[1] - physt[0]
frequency_Hz = np.fft.fftfreq(len(physt), d=dt)


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

# Extract physical magnetic-moment evolution
Mx_phys = Mx_t[repeatsum:]
My_phys = My_t[repeatsum:]
Mz_phys = Mz_t[repeatsum:]

# Thin physical evolution for responsive animation
plot_stride = 3

Mx_plot = Mx_phys[::plot_stride]
My_plot = My_phys[::plot_stride]
Mz_plot = Mz_phys[::plot_stride]
t_plot = physt[::plot_stride]

# Retain the artificial introductory hold frames
Mx_final = np.concatenate((Mx_t[:repeatsum], Mx_plot))
My_final = np.concatenate((My_t[:repeatsum], My_plot))
Mz_final = np.concatenate((Mz_t[:repeatsum], Mz_plot))

# Introductory hold frames are displayed at t = 0
t_final = np.concatenate((np.zeros(repeatsum), t_plot))

# Combined magnetic-moment history used for scene limits
M_final = np.column_stack((Mx_final, My_final, Mz_final))

# Transverse radius at each animation frame
r_perp_final = np.sqrt(Mx_final**2 + My_final**2)

# Center the FFT frequency bins and Fourier coefficients
frequency_plot = np.fft.fftshift(frequency_Hz)

FT_Sx_plot = np.abs(np.fft.fftshift(FT_Sx_t))
FT_Sy_plot = np.abs(np.fft.fftshift(FT_Sy_t))
FT_Sz_plot = np.abs(np.fft.fftshift(FT_Sz_t))

# Plot colors
page_background = "#06182B"
panel_background = "#123B5D"
axis_gold = "#FFCC00"
text_white = "#F5F7FA"
vector_red = "#FF4D4D"
circle_white = "#FFFFFF"
sx_red = "#FF6B6B"
sy_cyan = "#4DDFFF"
sz_green = "#39FF14"
grid_white = "rgba(255,255,255,0.18)"

# Animation controls
frame_duration_ms = 15

# Points used to construct the transverse-radius circle
theta = np.linspace(0.0, 2.0 * np.pi, 120)

# Number of animation frames
num_plot_frames = len(t_final)

# Equal limits for all three dimensions
max_abs_moment = np.max(np.abs(M_final))

if max_abs_moment == 0:
    axis_limit = 1.0
else:
    axis_limit = 1.10 * max_abs_moment

# Fourier-magnitude vertical-axis limit
max_fourier_magnitude = np.max(np.concatenate((FT_Sx_plot, FT_Sy_plot, FT_Sz_plot)))

if max_fourier_magnitude == 0:
    fourier_y_limit = 1.0
else:
    fourier_y_limit = 1.10 * max_fourier_magnitude

# Construct the title shown during each animation stage
def get_frame_title(k):
    if k < frame1repeats:
        return f"Thermal equilibrium → hold frame {k + 1} of {frame1repeats}"

    if k < repeatsum:
        return f"Immediately after RF pulse → hold frame {k - frame1repeats + 1} of {frame2repeats}"

    return f"Laboratory-frame Zeeman evolution → t = {t_final[k]:.3e} s"

# Construct the transverse circle at the current longitudinal position
def get_circle_coordinates(k):
    current_radius = r_perp_final[k]
    circle_x = current_radius * np.cos(theta)
    circle_y = current_radius * np.sin(theta)
    circle_z = np.full(theta.shape, Mz_final[k])

    return circle_x, circle_y, circle_z

# Create one three-dimensional animation panel and one Fourier-spectrum panel
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    column_widths=[0.56, 0.44],
    horizontal_spacing=0.08,
    subplot_titles=("Laboratory-Frame Precession", "Fourier Transform of Spin Expectations")
)

# Initial transverse-radius circle
circle_x_initial, circle_y_initial, circle_z_initial = get_circle_coordinates(0)

# Trace 0: instantaneous voxel magnetic-moment vector
fig.add_trace(
    go.Scatter3d(
        x=[0.0, Mx_final[0]],
        y=[0.0, My_final[0]],
        z=[0.0, Mz_final[0]],
        mode="lines+markers",
        line=dict(color=vector_red, width=8),
        marker=dict(size=[2, 6], color=vector_red),
        name="Magnetic-moment vector",
        showlegend=False,
        hovertemplate="Mₓ = %{x:.3e} A·m²<br>Mᵧ = %{y:.3e} A·m²<br>M_z = %{z:.3e} A·m²<extra></extra>"
    ),
    row=1,
    col=1
)

# Trace 1: instantaneous transverse-radius circle
fig.add_trace(
    go.Scatter3d(
        x=circle_x_initial,
        y=circle_y_initial,
        z=circle_z_initial,
        mode="lines",
        line=dict(color=circle_white, width=6),
        name="Transverse-radius circle",
        showlegend=False,
        hoverinfo="skip"
    ),
    row=1,
    col=1
)

# Trace 2: magnitude of the Fourier transform of <Sx>
fig.add_trace(
    go.Scatter(
        x=frequency_plot,
        y=FT_Sx_plot,
        mode="lines",
        line=dict(color=sx_red, width=3),
        name="|FT{⟨Sₓ⟩}|",
        hovertemplate="Frequency = %{x:.4e} Hz<br>|FT| = %{y:.4e}<extra></extra>"
    ),
    row=1,
    col=2
)

# Trace 3: magnitude of the Fourier transform of <Sy>
fig.add_trace(
    go.Scatter(
        x=frequency_plot,
        y=FT_Sy_plot,
        mode="lines",
        line=dict(color=sy_cyan, width=2, dash="dash"),
        name="|FT{⟨Sᵧ⟩}|",
        hovertemplate="Frequency = %{x:.4e} Hz<br>|FT| = %{y:.4e}<extra></extra>"
    ),
    row=1,
    col=2
)

# Trace 4: magnitude of the Fourier transform of <Sz>
fig.add_trace(
    go.Scatter(
        x=frequency_plot,
        y=FT_Sz_plot,
        mode="lines",
        line=dict(color=sz_green, width=3),
        name="|FT{⟨S_z⟩}|",
        hovertemplate="Frequency = %{x:.4e} Hz<br>|FT| = %{y:.4e}<extra></extra>"
    ),
    row=1,
    col=2
)

# Construct animation frames
animation_frames = []

for k in tqdm(range(num_plot_frames), desc="Building animation frames", unit="frame", dynamic_ncols=True):
    circle_x, circle_y, circle_z = get_circle_coordinates(k)

    current_frame = go.Frame(
        name=str(k),
        data=[
            go.Scatter3d(
                x=[0.0, Mx_final[k]],
                y=[0.0, My_final[k]],
                z=[0.0, Mz_final[k]]
            ),
            go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=circle_z
            )
        ],
        traces=[0, 1],
        layout=go.Layout(
            title=dict(text=get_frame_title(k))
        )
    )

    animation_frames.append(current_frame)

print("Building plot environment...")

# Attach animation frames
fig.frames = animation_frames

# Construct slider steps
slider_steps = []

for k in range(num_plot_frames):
    slider_step = dict(
        method="animate",
        args=[
            [str(k)],
            {
                "mode": "immediate",
                "frame": {
                    "duration": 0,
                    "redraw": True
                },
                "transition": {
                    "duration": 0
                }
            }
        ],
        label=""
    )

    slider_steps.append(slider_step)

# Style the complete figure
fig.update_layout(
    title=dict(
        text=get_frame_title(0),
        x=0.5,
        xanchor="center",
        font=dict(size=22, color=text_white)
    ),
    height=760,
    paper_bgcolor=page_background,
    plot_bgcolor=panel_background,
    font=dict(color=text_white),
    uirevision="constant",
    margin=dict(l=30, r=30, t=105, b=130)
)

# Style the three-dimensional laboratory-frame scene
fig.update_layout(
    scene=dict(
        bgcolor=panel_background,
        aspectmode="cube",
        dragmode="orbit",
        uirevision="constant",
        camera=dict(
            eye=dict(x=1.45, y=-1.0, z=0.8)
        ),
        xaxis=dict(
            title="Mₓ (A·m²)",
            color=axis_gold,
            backgroundcolor=panel_background,
            gridcolor=grid_white,
            zerolinecolor=axis_gold,
            linecolor=axis_gold,
            range=[-axis_limit, axis_limit],
            tickformat=".1e",
            showbackground=True
        ),
        yaxis=dict(
            title="Mᵧ (A·m²)",
            color=axis_gold,
            backgroundcolor=panel_background,
            gridcolor=grid_white,
            zerolinecolor=axis_gold,
            linecolor=axis_gold,
            range=[-axis_limit, axis_limit],
            tickformat=".1e",
            showbackground=True
        ),
        zaxis=dict(
            title="M_z (A·m²)",
            color=axis_gold,
            backgroundcolor=panel_background,
            gridcolor=grid_white,
            zerolinecolor=axis_gold,
            linecolor=axis_gold,
            range=[-axis_limit, axis_limit],
            tickformat=".1e",
            showbackground=True
        )
    )
)

# Style the Fourier-transform axes
fig.update_layout(
    xaxis=dict(
        title="Frequency (Hz)",
        color=axis_gold,
        linecolor=axis_gold,
        gridcolor=grid_white,
        zerolinecolor="rgba(255,255,255,0.35)",
        showline=True,
        mirror=True,
        tickformat=".2e",
        range=[frequency_plot[0], frequency_plot[-1]]
    ),
    yaxis=dict(
        title="Fourier Magnitude",
        color=axis_gold,
        linecolor=axis_gold,
        gridcolor=grid_white,
        zeroline=False,
        showline=True,
        mirror=True,
        tickformat=".2e",
        range=[0.0, fourier_y_limit]
    )
)

# Style subplot titles
fig.update_annotations(
    font=dict(color=axis_gold, size=17)
)

# Position the Fourier-transform legend within the right panel
fig.update_layout(
    legend=dict(
        x=0.99,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(6,24,43,0.88)",
        bordercolor=axis_gold,
        borderwidth=1,
        font=dict(color=text_white)
    )
)

# Add play and pause controls
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            showactive=False,
            x=0.43,
            y=-0.20,
            xanchor="center",
            yanchor="top",
            bgcolor=panel_background,
            bordercolor=axis_gold,
            font=dict(color=text_white),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "fromcurrent": True,
                            "mode": "immediate",
                            "frame": {
                                "duration": frame_duration_ms,
                                "redraw": True
                            },
                            "transition": {
                                "duration": 0
                            }
                        }
                    ]
                ),
                dict(
                    label="❚❚ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {
                                "duration": 0,
                                "redraw": False
                            },
                            "transition": {
                                "duration": 0
                            }
                        }
                    ]
                )
            ]
        )
    ]
)

# Add draggable animation slider
fig.update_layout(
    sliders=[
        dict(
            active=0,
            x=0.10,
            len=0.82,
            y=-0.055,
            xanchor="left",
            yanchor="top",
            bgcolor=panel_background,
            bordercolor=axis_gold,
            activebgcolor=sz_green,
            tickcolor=axis_gold,
            font=dict(color=text_white),
            currentvalue=dict(visible=False),
            pad=dict(t=30, b=0),
            steps=slider_steps
        )
    ]
)

# Add a gold border around the three-dimensional panel
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
    layer="above"
)

print("Writing interactive HTML file...")

# Open the interactive animation in the default browser
fig.write_html(
    "Hamiltonian_Bloch_Precession_Simulator.html",
    auto_open=True,
    auto_play=False,
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True
    },
    post_script="document.title = 'Hamiltonian Bloch Precession Simulator';"
)

print("HTML file complete.")