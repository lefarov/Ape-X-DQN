while getopts f:n: flag
	do
	    case "${flag}" in
	        f) filename=${OPTARG};;
	        n) nprocess=${OPTARG};;
	    esac
	done

mpirun -np $nprocess python $filename
