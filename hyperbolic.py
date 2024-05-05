# Based off of code by Anastasiia Tsvietkova, Dale Koenig & Alex Lowen.

import ast
from spherogram import links
import numpy as np
import sympy

BLACK = "Black"
WHITE = "White"

OPPOSITE = {WHITE: BLACK, BLACK: WHITE}


def k(edge):
    # MUST CHECK if edge goes left to right as viewed by region or right to left.
    # currently assumes left to right
    if edge.strand_index in [1, 3] and edge.opposite().strand_index in [0, 2]:
        return -1
    elif edge.strand_index in [0, 2] and edge.opposite().strand_index in [1, 3]:
        return 1
    else:
        return 0


def kappa(edge_1, edge_2):
    """This is just an EQUAL for existance in oriented_edges."""

    edge_1_oriented = any(edge_1.strand_index == head for head, tail in edge_1.crossing.directions)
    edge_2_oriented = any(edge_2.strand_index == head for head, tail in edge_2.crossing.directions)
    return 1 if edge_1_oriented == edge_2_oriented else -1


def get_f_n(zetas, offset):
    """See Page 4 of https://arxiv.org/pdf/1108.0510."""
    # Don't forget that the papers shape parameters are 1--indexed while our lists are 0--indexed.
    n = len(zetas)
    assert n >= 3
    assert offset in {-1, 0, 1}
    if n == 3:
        return 1 - zetas[1 + offset]

    f = 1 - zetas[1 + offset]
    g = 1 - zetas[1 + offset] - zetas[2 + offset]
    # We start with f = f_3, g = f_4.
    # We recursively build f_n = f_{n-1} - zeta_{n-1} f_{n-2}.
    for i in range(4, n):
        f, g = g, g - zetas[i - 1 + offset] * f
        # Now f = f_{i}, g = f_{i+1}.

    return g


def solve(equations, crossing_vars, edge_vars, crossing_labels, edge_labels, num_iterations=150):
    """Find a solution to the given system of equations.

    Uses Newton--Raphson iteration.
    It feels like we should be able to replace this with something like:
      scipy.optimize.root(equations, labels, jac=Df)
    """
    f_temp = sympy.lambdify(crossing_vars + edge_vars, equations)
    Jac = sympy.Matrix(equations).jacobian(crossing_vars + edge_vars)
    Df_temp = sympy.lambdify(crossing_vars + edge_vars, Jac)
    f = lambda x: f_temp(*x)
    Df = lambda x: Df_temp(*x)
    # Merge.
    labels = np.array(crossing_labels + edge_labels, dtype=np.complex_)
    for _ in range(num_iterations):
        f_x = f(labels)
        Df_x = Df(labels)
        q, r = np.linalg.qr(Df_x)
        incx = np.linalg.solve(r, -np.matmul(q.conj().T, f_x))
        labels = np.around(labels + incx, 6)

    # Extract.
    crossing_solutions, edge_solutions = labels[: len(crossing_labels)], labels[len(crossing_labels) : len(crossing_labels) + len(edge_labels)]
    return crossing_solutions, edge_solutions


def analyze(L):
    faces = [[x.opposite() for x in face] for face in L.faces()]  # This orientation matters since it defines clockwise in each region.
    crossings = L.crossings

    # Crossing variables and starting labels:
    crossing_vars = sympy.symbols([f"w{i}" for i in range(len(crossings))])
    crossing_labels = [0 + 0.5j if crossing.sign == 1 else 0 - 0.5j for crossing in crossings]

    # Edge variables and starting labels:

    # First checkerboard color all the faces.
    coloring = [None] * len(faces)
    coloring[0] = BLACK
    to_add = [0]  # A stack for DFT.
    while to_add:
        face_i = to_add.pop()
        for nbr_i, nbr in enumerate(faces):
            if coloring[nbr_i] is None and any(e.opposite() in faces[face_i] for e in nbr):
                coloring[nbr_i] = OPPOSITE[coloring[face_i]]
                to_add.append(nbr_i)

    # Iterate through black faces and assign indices to the corresponding edge.
    # At the same time, we assign indices and names to the reversed edge (bordering a white region)
    black_faces = [face for face, color in zip(faces, coloring) if color == BLACK]
    edge_indices = {side: index for index, side in enumerate(oedge for face in black_faces for edge in face for oedge in [edge, edge.opposite()])}

    edge_vars = sympy.symbols([f"{prefix}{i+1}" for i in range(2 * len(crossings)) for prefix in ["u", "v"]])
    edge_labels = [0] * 4 * len(crossings)

    equations = []
    # Now lets go through each face and create the equations
    for face in black_faces:
        for edge in face:
            edge_labels[edge_indices[edge]] = -0.5 - 0.5j  # initial edge label value for black side
            edge_labels[edge_indices[edge.opposite()]] = 0.5 - 0.5j  # initial edge label value for white side
            equations.append(edge_vars[edge_indices[edge]] - edge_vars[edge_indices[edge.opposite()]] - k(edge))

    for face in faces:
        if len(face) == 1:
            raise RuntimeError("Monogon")
        elif len(face) == 2:  # Bigon.
            e1, e2 = face
            equations.append(edge_vars[edge_indices[e1]])  # Enforce edge_label == 0
            equations.append(edge_vars[edge_indices[e2]])
            equations.append(crossing_vars[e1[0].label] - crossing_vars[e2[0].label])  # Crossings must be equal.
        else:  # len(face) >= 3:  # Generate equations for faces with more than 2 sides.
            shape_params = [
                kappa(prev_edge, edge)
                * crossing_vars[edge[0].label]
                / edge_vars[edge_indices[edge]]
                / edge_vars[edge_indices[prev_edge]]
                for edge, prev_edge in zip(face, face[-1:] + face[:-1])
            ]

            f_n = [get_f_n(shape_params, offset) for offset in [-1, 0, 1]]
            # The f_n functions have fractions, multiply through to clear denominators.
            f_n = [sympy.simplify(sympy.fraction(sympy.together(f))[0]) for f in f_n]  # The [0] here is selecting the numerator.

            equations.extend(f_n)

    print("Solve the following system of equations:")
    for eq in equations:
        print(f"\t{eq} = 0")

    crossing_solutions, edge_solutions = solve(equations, crossing_vars, edge_vars, crossing_labels, edge_labels)

    # Output time.
    print("Faces:")
    for face, color in zip(faces, coloring):
        print(f"{color} face: " + " -> ".join(str(edge[0][0]) for edge in face) + " ->")

    print("'u' labels correspond to the black sides of the edges.")
    print("'v' labels to the white sides.")
    print("'w' labels correspond crossings.")

    print("Calculated values:")
    for index, label in enumerate(crossing_solutions):
        print(f"w{index} = {label}")

    count = 0
    for index, face in enumerate(black_faces):
        for edge in face:
            print(f"u{count + 1} = {edge_solutions[count * 2]}")
            print(f"v{count + 1} = {edge_solutions[count * 2 + 1]}")
            print(f"\t(From edge {edge[0]} -> {edge.opposite()[0]})")
            count += 1


if __name__ == "__main__":
    print("Input a link by PD (Planar Diagram) code.  See, for example, https://arxiv.org/abs/1309.3288.")
    print("Note, with this input method it seems like spherogram may not accept signs in the input, so all links will be alternating.")
    print("Crossing and edge labels should start at 0.")
    print("Example input: '(1, 7, 2, 6), (5, 3, 6, 2), (7, 4, 0, 5), (3, 0, 4, 1)'")
    print("From this input, we know that there are 4 vertices labelled 0, 1, 2, and 3")
    print("We can draw the knot starting with the edge labelled 0, which connects crossing 2 and 3")
    print("The path from crossing 3 to crossing 0 with label 1, then on to crossing 1 with label 2, then to crossing 3 with label 3, etc.")
    print("We can see that crossing 0 will end up surrounded by edges labelled 1, 7, 2, an 6, precisely the first tuple given.\n")
    print(
        "Alternatively, Dowker-Thistlethwaite code can be used. See for example, https://en.wikipedia.org/wiki/Dowker%E2%80%93Thistlethwaite_notation."
    )
    print("Example input: 'DT: [(4,6,8,2)]'")
    print("")
    print("Input a link code:")

    # temp_inp = input()
    temp_inp = "DT: [(4,6,8,2)]"
    temp_inp = "DT: [(6, -10, -14, 12, -16, -2, 18, -4, -8)]"
    if temp_inp.startswith("DT: "):
        L = links.Link(temp_inp)
        print(f"Analyzing link with DT code {L.DT_code()}.")
    else:
        L = links.Link(ast.literal_eval(temp_inp))
        print(f"Analyzing link with PD code {L.PD_code()}")

    analyze(L)
