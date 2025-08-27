#include<algorithm>
#include<vector>
#include<cmath>

std::vector<fastjet::PseudoJet> remove_particles_from_clustering(std::vector<fastjet::PseudoJet> particles, std::vector<int> pruned_idxs){
    // Initialize copy of particles
    std::vector<fastjet::PseudoJet> pruned_particles;

    // Loop through particles and prune if needed
    for (int i=0; i<particles.size(); i++){
        int current_idx = particles[i].user_index();
        // If current index is in the prune list, then continue
        if (std::find(pruned_idxs.begin(), pruned_idxs.end(), current_idx) != pruned_idxs.end() ) {
            //float px = particles[i].px();
            //float py = particles[i].py();
            //std::cout << "pT: " << std::sqrt(px*px+py*py) << std::endl;
            continue;
        }
        // Otherwise add particle to the pruned array
        else {
            pruned_particles.push_back(particles[i]);
        }
    }
    return pruned_particles;
}
