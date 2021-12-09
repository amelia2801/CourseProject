echo "===================> Executing 0_prepareMovieActorsFile.py"
python 0_prepareMovieActorsFile.py > logs/prepareMovieActorsFile.log
##
status=$?
if test $status -eq 0
then
	echo "0_prepareMovieActorsFile Completed Successfully"
else
	echo "0_prepareMovieActorsFile FAILED .."
    exit status
fi
#################################################
#
echo "===================> Executing 1_ScrapeMoviesActors.py"
python 1_ScrapeMoviesActors.py > logs/ScrapeMoviesActors.log
##
status=$?
if test $status -eq 0
then
	echo "1_ScrapeMoviesActors.py Completed Successfully"
else
	echo "1_ScrapeMoviesActors.py FAILED .."
    exit status
fi
#################################################
#
echo "===================> Executing 2_ScrapeMoviesTags.py"
python 2_ScrapeMoviesTags.py > logs/ScrapeMoviesTags.log
##
status=$?
if test $status -eq 0
then
	echo "2_ScrapeMoviesTags.py Completed Successfully"
else
	echo "2_ScrapeMoviesTags.py FAILED .."
    exit status
fi
#################################################
#
echo "===================> Executing 4_buildCorpus.py"
python 4_buildCorpus.py > logs/buildCorpus.log
##
status=$?
if test $status -eq 0
then
	echo "4_buildCorpus.py Completed Successfully"
else
	echo "4_buildCorpus.py FAILED .."
    exit status
fi
#################################################
cd ../
#################################################
#
echo "===================> Executing buildSearchIndex.py"
python buildSearchIndex.py > logs/buildSearchIndex.log
##
status=$?
if test $status -eq 0
then
	echo "buildSearchIndex.py Completed Successfully"
else
	echo "buildSearchIndex.py FAILED .."
    exit status
fi