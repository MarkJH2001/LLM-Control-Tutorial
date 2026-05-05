# Aircraft Attitude Control

## System description

Consider a modern aircraft (Figure 1). The purpose of the aircraft considered here is to control the positions of the fins. Due to the requirements of improved reliability and response, the surfaces of modern aircraft are controlled by electric actuators with electronic controls.

Figure 1 illustrates the controlled surfaces and the block diagram of one axis of such a position-control system. Figure 2 shows the transfer function block diagram of the system. The system is simplified to the extent that saturation of the amplifier gain and motor torque, gear backlash, and shaft compliances have all been neglected. The objective of the system is to have the output, $\theta_y(t)$, follow the input, $\theta_r(t)$.

![Figure 1: Block diagram of an attitude control system of an aircraft](assets/block_diagram.png){: style="max-width:720px;" }

**Figure 1**: Block diagram of an attitude control system of an aircraft.

![Figure 2: Transfer function block diagram of the aircraft attitude control system](assets/tf_block_diagram.png){: style="max-width:720px;" }

**Figure 2**: Transfer function block diagram of the aircraft attitude control system.

## System parameters

The following system parameters are given initially:

| Parameter | Symbol | Value |
|---|---|---|
| Gain of encoder | $K_s$ | $1\;\mathrm{V/rad}$ |
| Gain of preamplifier | $K$ | $\mathrm{adjustable}$ |
| Gain of power amplifier | $K_1$ | $10\;\mathrm{V/V}$ |
| Gain of current feedback | $K_2$ | $0.5\;\mathrm{V/A}$ |
| Gain of tachometer feedback | $K_t$ | $0\;\mathrm{V/(rad/s)}$ |
| Armature resistance of motor | $R_a$ | $5.0\;\Omega$ |
| Armature inductance of motor | $L_a$ | $0.003\;\mathrm{H}$ |
| Torque constant of motor | $K_i$ | $9.0\;\mathrm{N \cdot m/A}$ |
| Back-EMF constant of motor | $K_b$ | $0.0636\;\mathrm{V/rad/s}$ |
| Inertia of motor rotor | $J_m$ | $0.0001\;\mathrm{kg \cdot m^2}$ |
| Inertia of load | $J_L$ | $0.01\;\mathrm{kg \cdot m^2}$ |
| Viscous-friction coefficient of motor | $B_m$ | $0.005\;\mathrm{N \cdot m \cdot s}$ |
| Viscous-friction coefficient of load | $B_L$ | $1.0\;\mathrm{N \cdot m \cdot s}$ |
| Gear-train ratio between motor and load | $N = \theta_y / \theta_m$ | $1/10$ |

Because the motor shaft is coupled to the load through a gear train with ratio $N$, $\theta_y = N\theta_m$, the total inertia and viscous-friction coefficient seen by the motor are
$$
J_t = J_m + N^2 J_L, \qquad B_t = B_m + N^2 B_L,
$$
and the gain of preamplifier is set to $K = 181.17$.

## Questions

### (1) Simulation and analysis

Simulate the proposed control system with MATLAB/Simulink/Python, and analyze the stability and performance of the system.

### (2) PD controller — root-locus design

Suppose a PD controller $G_c(s) = K_P + K_D s$ with $K_P = 1$ is applied. If different $K_D$ is selected, analyze performance of the control system by using root locus design and determine whether the following time-domain performance specifications can be satisfied:

- Steady-state error due to unit-ramp input $\le 0.000443$
- Maximum overshoot $\le 5\%$
- Rise time $t_r \le 0.005\;\mathrm{sec}$
- Setting time $t_s \le 0.005\;\mathrm{sec}$

### (3) PI controller — root-locus design

Suppose a PI controller $G_c(s) = K_P + \dfrac{K_I}{s}$ is applied. Determine $K_P$ and $K_I$ by using root locus design to satisfy the following time-domain performance specifications:

- Steady-state error due to acceleration input $\le 0.2$ (acceleration input is $\frac{1}{2}t^2$)
- Maximum overshoot $\le 5\%$
- Rise time $t_r \le 0.01\;\mathrm{sec}$
- Setting time $t_s \le 0.02\;\mathrm{sec}$

### (4) PID controller — root-locus design

Suppose a PID controller $G_c(s) = K_P + K_D s + \dfrac{K_I}{s}$. Determine $K_P$, $K_I$ and $K_D$ by using root locus design to satisfy the following time-domain performance specifications:

- Steady-state error due to acceleration input $\le 0.2$ (acceleration input is $\frac{1}{2}t^2$)
- Maximum overshoot $\le 5\%$
- Rise time $t_r \le 0.005\;\mathrm{sec}$
- Setting time $t_s \le 0.005\;\mathrm{sec}$

### (5) PID — frequency-domain design

Determine $K_P$, $K_I$ and $K_D$ by using frequency-domain design to satisfy the following frequency-domain performance specifications:

- Steady-state error due to acceleration input $\le 0.2$ (acceleration input is $\frac{1}{2}t^2$)
- Phase margin $\ge 70°$
- Resonant peak $M_r \le 1.1$
- Bandwidth $BW \ge 1000\;\mathrm{rad/sec}$

### (6) State-space model and state-feedback controller

Establish state-space model of the aircraft attitude control model and design a state feedback controller to improve the performance. Simulate the proposed control system with stability and performance analysis.

### (7) Robustness

For the time-domain design in (2), (3), (4), (6), when parameter uncertainty and modeling error are taken into account, simulate the proposed system and analyze its performance. Moreover, propose robust method to reduce the performance influences of parameter uncertainty and modeling error, and simulate the proposed robust method with Matlab/Simulink/Python.
