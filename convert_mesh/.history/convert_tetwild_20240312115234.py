def read_msh(filename):
    vertices = []
    tetrahedra = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        vertex_mode = False
        tetrahedron_mode = False
        for line in lines:
            if 'Vertices' in line:
                vertex_mode = True
                tetrahedron_mode = False
                continue
            elif 'Tetrahedra' in line:
                vertex_mode = False
                tetrahedron_mode = True
                continue
            
            if vertex_mode and line.strip().isdigit() == False:
                vertices.append(line.strip().split())
            elif tetrahedron_mode and line.strip().isdigit() == False:
                # Assuming the tetrahedron indices in the msh file are 1-based indexing
                # Adjust them to 0-based for the mesh format specification
                tet = [str(int(i) - 1) for i in line.strip().split()]
                tetrahedra.append(tet)
                
    return vertices, tetrahedra

def write_mesh(vertices, tetrahedra, filename):
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f'v {" ".join(v)}\n')
        for t in tetrahedra:
            file.write(f't {" ".join(t)}\n')

def convert_msh_to_mesh(msh_filename, mesh_filename):
    vertices, tetrahedra = read_msh(msh_filename)
    write_mesh(vertices, tetrahedra, mesh_filename)

# Example usage
msh_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.msh'
mesh_filename = '/Users/liujunyu/Downloads/ftetwild_output_msh/32770.stl_0121.mesh'
convert_msh_to_mesh(msh_filename, mesh_filename)
