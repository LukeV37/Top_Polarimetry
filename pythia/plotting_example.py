import uproot # library used for reading file I/O 
import matplotlib.pyplot as plt # library used for plotting
import numpy as np # library used to handle vectors

# Read data from file and store into local variables
with uproot.open("data.root:data") as f:
    # Read in input variables (features)
    jet_pt = f['jet_pt'].array()
    jet_eta = f['jet_eta'].array()
    jet_phi = f['jet_phi'].array()
    jet_m = f['jet_m'].array()
    trk_pt = f['trk_pt'].array()
    trk_eta = f['trk_eta'].array()
    trk_phi = f['trk_phi'].array()
    trk_m = f['trk_m'].array()
    trk_q = f['trk_q'].array()
    trk_d0 = f['trk_d0'].array()
    trk_z0 = f['trk_z0'].array()
    trk_origin =  f['trk_origin'].array()
    # Read in the output variable (target)
    cos_theta = f['costheta'].array()

# Print general dataset information
print("Number of Events Simulated: ", len(jet_pt))           # Events are "flat" vector. No need to flatten.
print("Number of Jets Simulated: ", len(np.ravel(jet_pt)))   # Jets are "jagged" vector. Must flatten (ravel)!
print("Number of Tracks Simulated: ", len(np.ravel(trk_pt))) # Tracks are "jagged" vector. Must flatten (ravel)!
print() # Print empty line

# Print some data from first few events
num_events = 2

# Loop over events
for event in range(num_events):

    # Print event number
    print("Event Number: ", event)

    # Loop over each jet in event
    num_jets = len(jet_pt[event])
    for jet in range(num_jets):
        # Print some jet info
        print("\tJet Number: ", jet, "has a pT of: ", jet_pt[event][jet])

    print() # Print empty line

    # Loop over each trk in event
    num_trks = len(trk_pt[event])
    for trk in range(num_trks):
        # Print some trk info
        print("\tTrk Number: ", trk, "has a pT of: ", trk_pt[event][trk])

    print() # Print empty line

# Plot the data (Note: vectors must be flattened (raveled) before plotting!)
plt.hist(np.ravel(jet_pt),label='jet_pT', bins=20, range=(200,1000),histtype='step',color='r')
plt.title("Jet Transverse Momentum (pT)")
plt.ylabel('Counts')
plt.xlabel('pT (GeV)')
plt.yscale('log')
plt.legend()
plt.show()                 # Plot figure directly to display
#plt.savefig("jet_pt.png") # Optional line to save figure as png
