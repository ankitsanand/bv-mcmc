cd job_search/30
echo "Running Job Search Domain"
echo "Variant 30"
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 30"
echo "***************************************"

cd ../50
echo "Variant 50"
python construct_graph_orbital.py test test_orbital.saucy
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 50"
echo "***************************************"

cd ../50_10evid
echo "Variant 50_10evid"
python construct_graph_orbital.py test test_orbital.saucy
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 50_10evid"
echo "***************************************"

echo "######################################"
echo "Done with Job Search Domain"
echo "######################################"

cd ../../student_curriculum/600_50
echo "Running Student Curriculum Domain"
echo "Variant 600_50"
python construct_graph_orbital.py test test_orbital.saucy
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 600_50"
echo "***************************************"

cd ../1200_100
echo "Variant 1200_100"
python construct_graph_orbital.py test test_orbital.saucy
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 1200_100"
echo "***************************************"

cd ../1200_100_10evid
echo "Variant 1200_100_10evid"
python construct_graph_orbital.py test test_orbital.saucy
$GAP_DIR/bin/gap.sh -q < orbital_gibbs.g
echo "Done with Variant 1200_100_10evid"
echo "***************************************"

