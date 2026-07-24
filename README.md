# Generalized Open-System Spin Simulator for MRI

A density-matrix framework for simulating MRI spin dynamics using
Liouville–von Neumann and Lindblad master equations.

## Overview

This project models nuclear spin dynamics under Zeeman Hamiltonians,
RF excitation, longitudinal relaxation, and transverse dephasing.
Unlike a conventional Bloch-vector simulator, the framework evolves
density matrices and supports unitary dynamics, mixed states,
open-system relaxation channels, repeated-pulse steady states, and
spin phase-space representations.

## Current Simulations

1. Unitary, nonrelaxing Zeeman precession
2. Single-excitation T2 dephasing through Lindblad dynamics
3. Repeated RF excitations with T1 relaxation toward a steady state
4. Wigner and Husimi-Q phase-space representations of the single-pulse dynamics

## Methods

The closed-system dynamics follow the Liouville–von Neumann equation,

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho]
```

while relaxation and dephasing are modeled using Lindblad dissipators,

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho]
+
\sum_k
\left(
L_k \rho L_k^\dagger
-
\frac{1}{2}
\left\{
L_k^\dagger L_k,\rho
\right\}
\right)
```

## Development Status

The current framework focuses on spin-1/2 MRI dynamics. Planned extensions 
include numerical validation against established solvers and the incorporation 
of more sophisticated coupling and interaction effects. Longer-term directions 
include generalizing the open-system framework to weakly anharmonic superconducting
circuits such as transmon qubits.
