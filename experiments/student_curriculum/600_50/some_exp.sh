echo "*** Starting 0.01***"
sed -i '12s/1.0/0.01/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_01.csv
echo "### Ending 0.01 ###"

echo "*** Starting 0.02***"
sed -i '12s/0.01/0.02/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_02.csv
echo "### Ending 0.02 ###"

echo "*** Starting 0.05 ***"
sed -i '12s/0.02/0.05/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_05.csv
echo "### Ending 0.05 ###"

echo "*** Starting 0.10 ***"
sed -i '12s/0.05/0.10/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_10.csv
echo "### Ending 0.10 ###"

echo "*** Starting 0.25 ***"
sed -i '12s/0.20/0.25/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_25.csv
echo "### Ending 0.25 ###"

echo "*** Starting 0.50 ***"
sed -i '12s/0.25/0.50/' block_mcmc.g 
$GAP_DIR/bin/gap.sh -q < block_mcmc.g
mv block_gibbs.csv ./results/block_gibbs_50.csv
echo "### Ending 0.50 ###"

