# To do 22 Feb 2026

Working on testing PsiFormer vs Slater x BoseNet architectures. I should think about how to do a head to head comparison, and also think about other possible architecture designs.

I want to get setup on engaging and start submissing GPU jobs there.

There is still probably room for code performance improvement, but I should use AI agents to do thorough benchmarking, profiling, and improvement.

Could be worth it to try hidden fermion architectures too.

# 23 Feb 2026

Both PsiFormer and BoseNet appear to be struggling to get 27 site 1/3 FCI. VMC energy is around -13.62 and 1Band is about -13.65 for V1=0.3. Implemented vit for mote2 model now and going to run test job to see how it works.

It was the slater vit architecture, not boseformer, that basically fitted the exat ground state on 3x3 nu=1/3 Haldane model. Interesting.

Might be worth thinking about Hidden fermion/parton wavefunction ansatze.

# 24 Feb 2026

Systems to study:

    -FCI on lattice or MoTe2 with scalability
    -ACFL on lattice or MoTe2 with scalability
    -

Architectures
    -Hidden fermion/parton wavefunction
    -PsiFormer
    -BoseFormer x Slater
    -ViT x Slater
    -Message-Passing Neural Networks (MPNNs)

# 02 Mar 2026

Both PsiFormer and Slater x ViT are struggling to get good energy for 3 electrons nu=1/3. However, it seems that Slater x ViT does slightly better.

    -Should we study the square lattice Hofstadter model at quarter flux?
    -triangular lattice at quarter of 1/3 flux

The Slater x ViT architecture is working GREAT! I  made the attention kernel not depend only on the relative separation between two sites now, and this works much better.

# 03 Mar 2026

PsiFormer also seems to be able top get exact ground state for 3x3 and 4x3 Haldane model. Both types of coordinate embeddings (periodic and site index) seem to work similarly well as far as I can tell.