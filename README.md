# Finted Element Method Simulation

## Run the code and parameters settings

I implemented a pipeline to pass parameters using a ```.ini``` file. The absolute ```.ini``` file path is needed as the command line argument.

An example .ini file setting is shown as follows (The parameters are stored in a ```Setting``` struct in ```src/simulation.h```).

```
[IO]
    Mesh = ./example-meshes/ellipsoid.mesh       ; The dynamically simulated mesh path
    ObstacleMesh = ./example-meshes/sphere.mesh  ; The obstacle mesh path (optional)

[Transform]
	TranslationX = 0  ; Translation along X direction (for the dynamically simulated mesh)
	TranslationY = 2  ; Translation along Y direction (for the dynamically simulated mesh)
	TranslationZ = 0  ; Translation along Z direction (for the dynamically simulated mesh)
	RotationZ = 0     ; Rotation along Z direction (for the dynamically simulated mesh)
  ; (Note that for other translations, it's intuitive and easy to add into this pipeline.)

[Settings]
    integrate_method = 3  ; 1: Euler, 2: Mid-point, 3: RK4
    g = -1                ; gravity (along Y direction)
    kFloor = 4e4          ; floor collision
    lambda = 4e3          ; strain param
    mu = 4e3              ; strain param
    phi = 100             ; stress param
    psi = 100             ; stress param
    density = 1200.0      ; density of the mesh

    isCustomizeTimeStep = 0      ; 1: Enable customized time step, 0: Use either the per-frame timesetp or the adaptive time step
    integrationTimeStep = 0.005  ; Customized time step
    isAdaptiveTimeStep = 0       ; 1: Use adpative time step, 0: Do not use adpative time step

    isParallelize = 0            ; 1: Enable parallelization, 0: Disable parallelization

    isFBO = 1                    ; 1: Enable FBO colorful background, 0: Disable FBO colorful background
    isFXAA = 0                   ; 1: Enable FXAA, 0: Disable FXAA
```

## Basic Requirements

### Demos

**Single Tet**

https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/2b114db4-c7ab-41cd-a643-9a46fd06ea4c

**Cube**

https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/5de222a5-a416-45b4-a21a-d6eff33296be

**Ellipsoid and sphere**

https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/b793d2fb-c065-48fe-a7ab-edb159f42c85

### Implemenetation details

**Extracting the surface mesh**: The general idea is to extraces the external boundary faces by distinguishing them from internal faces by their occurrence frequency. Specifically, the implementation identifies surface meshes within a tetrahedral mesh by iterating over each tetrahedron, generating keys for each face that account for permutations to identify unique faces, and ensuring correct face winding based on the tetrahedron's centroid direction. Faces appearing exactly once in this process are deemed to be part of the surface mesh, as internal faces will appear more than once due to being shared by adjacent tetrahedra. 

**Computing and applying internal forces**: The implementation computes and applies internal forces within a simulated shape by iterating over each tetrahedron, calculating deformation matrices for current and rest positions, and strain and stress tensors to determine elastic and damping forces. Elastic and damping stresses are derived from the deformation and velocity matrices related to the tetrahedron's vertices, accounting for material-specific constants. These stresses are then used to compute internal forces on each vertex of a tetrahedron, applying the principle of virtual work to convert stress into forces, which are summed up and applied to vertices, affecting the simulation's dynamics.

**Collision resolution**: The collision resolving implementation first handles collisions with a predefined ground plane by adjusting the vertex's position to be just above the ground if it's detected below it, and then recalculates the vertex's velocity to simulate bounce and friction effects based on a restitution coefficient and ground friction. For mesh collision, it iterates through each face of a static mesh, checking for intersections between the vertex's path (from its last position to its current one) and the mesh faces. If an intersection is found, the vertex position is adjusted to a point just above the collision face, and its velocity is recalculated similarly to the ground collision case, using the face's normal to determine the bounce direction and applying friction to the tangential velocity component.

**Explicit integration method**:
- The Euler integration method applies forces to vertices, updating their acceleration based on the inverse mass and force, then updates velocity and position linearly with respect to time. 
- The Midpoint integration method initially updates velocity and computes an intermediate position using only half of the delta time, effectively using the midpoint of the velocity for the final position update. Then, it recalculates forces and updates velocity and position for the full time step from the original position. 


## Extra Credits

### Cool Tet Mesh

I converted 2 cool tet meshes from **TetWild** and incorporated them into the simulation:
- The mesh processing code can be found in ```convert_mesh/```. I wrote several python scripts to convert the binary ```.msh``` format to ```.mesh``` format and scale-normalize the mesh.
- By running ```tetwild.ini```

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/07201622-f9d2-495f-a6f2-9f0fb2adddfb
  
- By running ```tetwild_2```

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/a6be7da5-17a4-48b0-babe-2782dd56ddf2

### Make the visualizer pretty 

I incorporate an FBO into the OpenGL pipeline:
- By setting ```isFBO = true```, the background (previously white) will be set to colorful gradient colors.
- By setting ```isFXAA = true```, the whole frame will be processed by fast antialiasing.
- Other FBO tricks like filter could be easily incorporated.

https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/f8b67c79-b758-4c3c-9cfb-b0ff028c8941

### A higher-order explicit integrator

I implemented Runge-Kutta 4 integration. Initially, it applies forces to determine initial accelerations (k1), then estimates midpoints (k2 and k3) and an endpoint (k4) for both velocities and positions, adjusting each vertex's state at each stage. Finally, it updates each vertex's velocity and position using a weighted average of these slopes, effectively balancing the initial and final accelerations with the midpoint estimates.

The following comparison use ```tetwild.ini```:
- When using **euler** integration with a fixed time step **0.005s**, the simulation would explode.
  
  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/dc094e32-a0ad-4f5b-b890-0e82c0617d6c
  
- When using **mid-point and Runge-Kutta 4** integration with a fixed time step **0.005s**, the simulation runs smoothly.
  
  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/de6a55f5-7499-4b25-b664-87c4ffa6ddda

- When using **mid-point** integration with a fixed time step **0.007s**, the simulation would explode.

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/cbdcbb76-5098-4819-a013-3fce7755bba8
  
- When using **Runge-Kutta 4** integration with a fixed time step **0.007s**, the simulation runs smoothly.

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/4a7f9769-a762-4020-bbd1-6e45469ab07a

### Adaptive time stepping 

I implemented adaptive time stepping. Starting with an initial time step, the simulation integrates the system's state forward in time. It then compares this result to one obtained by taking smaller steps that cumulatively cover the same time interval, calculating the error between these two approaches. If the error exceeds a predefined threshold, the time step size is reduced to improve accuracy; conversely, if the error is significantly below the threshold, the time step size is increased to speed up the simulation while still maintaining acceptable accuracy. 

The following comparisons use ```tetwild.ini```:
- When using euler integration with a **fixed time step 0.005s**, the simulation would explode.

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/67a68c6c-d471-4e3b-9621-cdf00cfc3873
  
- When using euler integration with **adaptive time step**, the simulation runs smoothly.

  https://github.com/brown-cs-224/fem-Junyu-Liu-Nate/assets/75256586/7df463e7-e88c-4ad9-8d83-12148532b7d5
  

### Parallelize your code 

I tried both the **Openmp** and **QtConcurrent** for parallelization:
- Both parallelization doesn't seem to improve the performance much. I suppose the main reason is that the meshes are not complex enough to show the effect of parallelization.
- I tried using extremely small time steps to stress the simulation, which does cause non-smooth simulation but still doesn't show much difference between parallelized and non-parallelized code. I suppose the main reason is that the using small time steps simply procudes more non-parallelizable operations (each step has to depend on the previous step).
- I tried converting more complex TetWild meshes using my script but found it tricky to scale-normalize the meshes, due to the complicated (highlt concave meshes with holes).
- Openmp parallelized code can be found in ```src/simulation.cpp/eulerIntegrateParallel```
- QtConcurrent parallelized code can be found in ```src/simulation.cpp/midPointIntegrateParallel``` and ```src/simulation.cpp/calculateMaxPositionErrorParallel```.



## Collaboration/References

I clarify that there is no collaboration include when I do this project.

References for mathematical formulas are coming from course slides.

## Known Bugs

The main bug is the non-obvious efficiency improvement in Parallelize your code part. I have explained my trials in that part in detail.
