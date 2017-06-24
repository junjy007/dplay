# RUNNING_DIR=../../RUNS
test_script=$(ls tests_and_notebooks/$1*.py 2>/dev/null)
n=$(echo $test_script | wc -w)

# rdir=$(cd $RUNNING_DIR; pwd)/$1
# read -r -p "Deploy $test_script to ${rdir} [y/n]? " response
# if [[ "$response" =~ ^([yY])+$ ]]
# then
# python dplay_utils/xdeploy.py deploy ${test_script}
# fi
 
if [ $n -eq 1 ]
then
	echo Deploying ${test_script}...
	python dplay_utils/xdeploy.py deploy ${test_script}
else
	echo Not found: unique experiment script: ${test_script} 
fi
