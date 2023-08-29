# Beam Analysis using Finite Element Method
# Developed by Engineer Hunter

# Updated door Edgars Labans


# units of EI GA and L, q should be synchronized
# werkt N/m EI 540721, GA = 510727  , L1 = 3, q_load = 1000\
# symetry for 3 spans is not correct


import numpy as np
import matplotlib.pyplot as plt


def calculate_beam():

    nmm = False
    # Structure Input
    EI: float = 779330      # N/mm2 (MPa)   *0.000001

    GA = 523520   #*1000 #*0.0001   # N

    E = EI
    I = 1

    t_skin = 12
    h_total = 224
    b_total = 1000

    overst = 2  # mm
    L1 = 4
    L2 = 4
    L3 = 0


    LP1 = 6
    LP2 = 9

    LP1_load = 666
    LP2_load = 999

    q_load = 1211      # kN/m    *0.001

    if nmm:
        EI = EI * 1000000
        L1 = L1 * 1000
        q_load = q_load * 0.001

    total_beam = overst + L1 + L2 + L3

    ref =q_load * (2 ** 4 / (24 * EI) - 2 ** 2 / (2 * GA) + 4 / 2 * (-2 ** 3 / (6 * EI) + 2 / GA) + 4 ** 3 / (24 * EI) * 2)

    print("ref", ref)

    ks = 1  # form factor

    # A = (h_total-2*t_skin)*b_total
    # I= (b_total * h_total^3)/12 - (b_total * (h_total - t_skin*2)^3)/12

    class Beam:
        def __init__(self, young, inertia, node, bar):
            self.young = young
            self.inertia = inertia
            self.node = node.astype(float)
            self.bar = bar.astype(int)

            self.dof = 2
            self.point_load = np.zeros_like(node)
            self.distributed_load = np.zeros([len(bar), 2])
            self.support = np.ones_like(node).astype(int)
            self.section = np.ones(len(bar))

            self.force = np.zeros([len(bar), 2 * self.dof])
            self.displacement = np.zeros([len(bar), 2 * self.dof])
            self.f = np.zeros(1)

        def analysis(self):
            nn = len(self.node)
            ne = len(self.bar)
            n_dof = self.dof * nn
            d = self.node[self.bar[:, 1], :] - self.node[self.bar[:, 0], :]
            length = np.sqrt((d ** 2).sum(axis=1))

            # Form Structural Stiffness
            matrix = np.zeros([2 * self.dof, 2 * self.dof])
            k = np.zeros([ne, 2 * self.dof, 2 * self.dof])
            ss = np.zeros([n_dof, n_dof])
            for i in range(ne):
                # Generate DOF
                aux = self.dof * self.bar[i, :]
                index = np.r_[aux[0]:aux[0] + self.dof, aux[1]:aux[1] + self.dof]
                # Element Stiffness Matrix
                l: float = length[i]
                fi = (12 * EI) / (ks * GA * l ** 2)

                matrix[0] = [12, 6 * l, -12, 6 * l]
                matrix[1] = [6 * l, (4 + fi) * l ** 2, -6 * l, (2 - fi) * l ** 2]
                matrix[2] = [-12, -6 * l, 12, -6 * l]
                matrix[3] = [6 * l, (2 - fi) * l ** 2, -6 * l, (4 + fi) * l ** 2]

                k[i] = (EI * matrix) / ((l ** 3) * (1 + fi))

                # Global Stiffness Matrix
                ss[np.ix_(index, index)] += k[i]

            # Distributed Load
            eq_load_ele = np.zeros([len(self.bar), 2 * self.dof])
            for i in range(len(self.bar)):
                l: float = length[i]
                pi: float = self.distributed_load[i, 0]
                pf: float = self.distributed_load[i, 1]
                eq_load_ele[i, 0] = l * (21 * pi + 9 * pf) / 60
                eq_load_ele[i, 1] = l * (l * (3 * pi + 2 * pf)) / 60
                eq_load_ele[i, 2] = l * (9 * pi + 21 * pf) / 60
                eq_load_ele[i, 3] = l * (l * (- 2 * pi - 3 * pf)) / 60

            # Point Load
            for i in range(len(self.bar)):
                self.point_load[self.bar[i, 0], 0] += eq_load_ele[i, 0]
                self.point_load[self.bar[i, 0], 1] += eq_load_ele[i, 1]
                self.point_load[self.bar[i, 1], 0] += eq_load_ele[i, 2]
                self.point_load[self.bar[i, 1], 1] += eq_load_ele[i, 3]

            # Solution
            free_dof = self.support.flatten().nonzero()[0]
            kff = ss[np.ix_(free_dof, free_dof)]
            p = self.point_load.flatten()
            pf = p[free_dof]
            uf = np.linalg.solve(kff, pf)
            u = self.support.astype(float).flatten()
            u[free_dof] = uf
            u = u.reshape(nn, self.dof)
            u_ele = np.concatenate((u[self.bar[:, 0]], u[self.bar[:, 1]]), axis=1)
            for i in range(ne):
                self.force[i] = np.dot(k[i], u_ele[i]) - eq_load_ele[i]
                self.displacement[i] = u_ele[i]

        def plot(self, deformed=False, scale=None, moment=False, shear=False, text=False):

            if np.amax(self.displacement) > 10:
                round_to_d = 0
            else:
                round_to_d = 4


            plt.rcParams.update({'font.family': 'Arial'})
            fig, axs = plt.subplots(4)
            if deformed is True:
                for i in range(len(self.bar)):
                    xi, xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    yi, yf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
                    axs[0].plot([xi, xf], [yi, yf], color='b', linestyle='-', linewidth=2)
                if scale is None:
                    scale = 1
                for i in range(len(self.bar)):
                    dxi, dxf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    dyi = self.node[self.bar[i, 0], 1] + self.displacement[i, 0] * scale
                    dyf = self.node[self.bar[i, 1], 1] + self.displacement[i, 2] * scale
                    axs[0].plot([dxi, dxf], [dyi, dyf], color='r', linestyle='--', linewidth=2)
                    if text is True:
                        axs[0].text(dxi, dyi, str(round(dyi / scale, round_to_d)), rotation=90)
                        if i == len(self.bar) - 1:
                            axs[0].text(dxf, dyf, str(round(dyf / scale, round_to_d)), rotation=90)
                axs[0].axis('off')
                axs[0].set_title('Deformation, m', y=-0.25,loc='left')

            if moment is True:
                axs[1].invert_yaxis()

                if np.amax(self.force) > 10:
                    round_to = 0
                else:
                    round_to = 4

                for i in range(len(self.bar)):
                    mxi, mxf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    myi, myf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
                    axs[1].plot([mxi, mxf], [myi, myf], color='b', linestyle='-', linewidth=1)
                for i in range(len(self.bar)):
                    ax, bx = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    ay, by = - self.force[i, 1], self.force[i, 3]
                    axs[1].plot([ax, ax, bx, bx], [0, ay, by, 0], color='g', linestyle='-', linewidth=1)
                    axs[1].fill([ax, ax, bx, bx], [0, ay, by, 0], 'c', alpha=0.3)
                    if text is True:
                        axs[1].text(ax, ay, str(round(ay, round_to)), rotation=90)
                        if i == len(self.bar) - 1:
                            axs[1].text(bx, by, str(round(by, round_to)), rotation=90)
                axs[1].axis('off')
                axs[1].set_title('Bending Moment Nm', y=-0.25,loc='left')
            if shear is True:
                for i in range(len(self.bar)):
                    sxi, sxf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    syi, syf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
                    axs[2].plot([sxi, sxf], [syi, syf], color='b', linestyle='-', linewidth=1)
                for i in range(len(self.bar)):
                    cx, dx = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    cy, dy = - self.force[i, 0], self.force[i, 2]
                    axs[2].plot([cx, cx, dx, dx], [0, cy, dy, 0], color='r', linestyle='-', linewidth=1)
                    axs[2].fill([cx, cx, dx, dx], [0, cy, dy, 0], 'orange', alpha=0.3)
                    if text is True:
                        axs[2].text(cx, cy, str(round(cy, round_to)), rotation=90)
                        if i == len(self.bar) - 1:
                            axs[2].text(dx, dy, str(round(dy, round_to)), rotation=90)
                axs[2].axis('off')
                axs[2].set_title('Shear Force, N', y=-0.25,loc='left')


            if shear is True:
                for i in range(len(self.bar)):
                    sxi, sxf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    syi, syf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
                    axs[3].plot([sxi, sxf], [syi, syf], color='b', linestyle='-', linewidth=1)
                for i in range(len(self.bar)):
                    cx, dx = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
                    cy, dy =  - self.distributed_load[i, 0], - self.distributed_load[i, 1]
                    axs[3].plot([cx, cx, dx, dx], [0, cy, dy, 0], color='r', linestyle='-', linewidth=1)
                    axs[3].fill([cx, cx, dx, dx], [0, cy, dy, 0], 'blue', alpha=0.3)

                axs[3].text(0, np.amax(-self.distributed_load), str(round(self.distributed_load[0, 0], round_to)), rotation=0)

                #axs[3].text(0, 0, str(round(dy, round_to)), rotation=90)


                axs[3].axis('off')
                axs[3].set_title('Loading, N', y=-0.25, loc='left')
                axs[3].plot(0, np.amax(-self.distributed_load)*2)


                if LP1_load > LP2_load:
                    len_LP1 =  np.amax(-self.distributed_load)*2
                    len_LP2 = (LP2_load/LP1_load) * np.amax(-self.distributed_load) + np.amax(-self.distributed_load)
                else:
                    len_LP2 =  np.amax(-self.distributed_load)*2
                    len_LP1 = (LP1_load/LP2_load) * np.amax(-self.distributed_load) + np.amax(-self.distributed_load)


                if LP1 > 0:
                    axs[3].arrow(LP1, len_LP1, 0, -len_LP1 + 250,  head_length = 300, head_width = 0.3,  width = 0.05, ec ='green')
                    axs[3].text(LP1, len_LP1, str(round(LP1_load, 0)), rotation=0)

                if LP2 > 0:
                    axs[3].arrow(LP2, len_LP2, 0, -len_LP2 + 250,  head_length = 300, head_width = 0.3,  width = 0.05, ec ='green')
                    axs[3].text(LP2, len_LP2, str(round(LP2_load, 0)), rotation=0)





    if LP1 > total_beam:
        LP1_load = 0
        LP1 = 0
        print("LP1 are disabled due to location beyond the beam length")

    if LP2 > total_beam:
        LP2_load = 0
        LP2 = 0
        print("LP2 are disabled due to location beyond the beam length")

    nodes = np.array([[0, 0],
                      [overst / 3, 0],  # LP1 - 1
                      [overst * 2 / 3, 0],  # LP2 - 2
                      [overst * 3 / 3, 0],

                      [overst + L1 * 1 / 4, 0],  # LP1 - 4
                      [overst + L1 * 2 / 4, 0],
                      [overst + L1 * 3 / 4, 0],  # LP2 - 6
                      [overst + L1 * 4 / 4, 0],

                      [overst + L1 + L2 * 1 / 4, 0],  # LP1 - 8
                      [overst + L1 + L2 * 2 / 4, 0],
                      [overst + L1 + L2 * 3 / 4, 0],  # LP2 -10
                      [overst + L1 + L2 * 4 / 4, 0],

                      [overst + L1 + L2 + L3 * 1 / 4, 0],  # LP1 -12
                      [overst + L1 + L2 + L3 * 2 / 4, 0],
                      [overst + L1 + L2 + L3 * 3 / 4, 0],  # LP1 -13
                      [overst + L1 + L2 + L3 * 4 / 4, 0]])



    # adding extra nodes for the point load
    nodes = np.vstack((nodes, np.array([LP1, 0])))
    nodes = np.vstack((nodes, np.array([LP2, 0])))

    # removing duplicates
    nodes = np.unique(nodes, axis=0)

    # sorting nodes
    ind = np.argsort(nodes, axis=0)
    nodes = np.take_along_axis(nodes, ind, axis=0)

    # joining nodes with bars
    bars = np.array([0, 1])

    for x in range(1, len(nodes) - 1):
        bars = np.vstack((bars, np.array([x, x + 1])))

    print(bars)

    # creating a global stiffness matrix
    beam_1 = Beam(E, I, nodes, bars)

    # Adding point load
    point_load_1 = beam_1.point_load
    point_load_1[np.where(nodes == LP1)[0][0], 0] = - LP1_load
    point_load_1[np.where(nodes == LP2)[0][0], 0] = - LP2_load

    # adding distributed load to the whole beam
    distributed_load_1 = beam_1.distributed_load
    for ind, bar in bars:
        distributed_load_1[ind] = np.array([-q_load, -q_load])

    support_1 = beam_1.support
    support_1[np.where(nodes == overst)[0][0], 0] = 0  # 0 - verplaatsing; 1- moment, : -alles
    support_1[np.where(nodes == overst + L1)[0][0], 0] = 0
    support_1[np.where(nodes == overst + L1 + L2)[0][0], 0] = 0
    support_1[np.where(nodes == overst + L1 + L2 + L3)[0][0], 0] = 0

    beam_1.analysis()
    np.set_printoptions(precision=5, suppress=True)
    #beam_1.plot(deformed=True, scale=500, moment=True, shear=True, text=True)

    print(beam_1.distributed_load)
    print(beam_1.force)
    print(beam_1.displacement)

    # Show plot
    #plt.show()

    return beam_1.displacement.max()
