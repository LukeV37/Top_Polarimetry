import uproot
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

with uproot.open("../WS_U_10k_test/dataset_U_10k_test_0.root:pythia") as f:
    print(f.keys())
    px = f["p_px"].array()
    py = f["p_py"].array()
    pz = f["p_pz"].array()
    pid = f["p_pid"].array()
    fromDown = f["p_fromDown"].array()
    fromUp = f["p_fromUp"].array()
    fromBottom = f["p_fromBottom"].array()
    fromLepton = f["p_fromLepton"].array()
    fromNu = f["p_fromNu"].array()
    fromAntiBottom = f["p_fromAntiBottom"].array()

for event in range(10):
    print("Processing Event No: ", event)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for particle in range(len(px[event])):
        
        if px[event][particle]**2+py[event][particle]**2<1:
            continue
        
        if fromDown[event][particle]==1:
            color='r'
            label='down'
        elif fromUp[event][particle]==1:
            color='chocolate'
            label='up'
        elif fromBottom[event][particle]==1:
            color='orange'
            label='bottom'
        elif fromLepton[event][particle]==1:
            color='y'
            label='lepton'
        elif fromNu[event][particle]==1:
            color='lawngreen'
            label='nu'
        elif fromAntiBottom[event][particle]==1:
            color='turquoise'
            label='anti-bottom'
        else:
            color='k'
            label='ISR/FSR'

        v = np.array([px[event][particle], py[event][particle], pz[event][particle]])
        vlength=np.linalg.norm(v)
        v_norm = v/vlength
        
        ax.quiver(0, 0, 0, v_norm[0], v_norm[1], v_norm[2], color=color, arrow_length_ratio=0.3/vlength, label=label)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Event No: '+str(event))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # 360 Degree view
    for angle in tqdm(range(0, 360)):
       ax.view_init(30, angle)
       plt.draw()
       plt.savefig(f"../WS_U_10k_test/output/event_{event}_angle_{angle}.png")
       #plt.pause(.001)

    #plt.show()
