echo "**** Running marginals_generator ****"
$GAP_DIR/bin/gap.sh -q < marginals_generator.g

echo "**** Running gibbs.g ****"
$GAP_DIR/bin/gap.sh -q < gibbs.g

echo "**** Running nc_gibbs.g ****"
$GAP_DIR/bin/gap.sh -q < nc_gibbs.g

echo "**** Running block_gibbs.g ****"
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
