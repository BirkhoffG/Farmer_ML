#git fetch --all
#git reset --hard origin/master
ZIPDIR=$(ls -t ./result | head -1)
#python 'DL Model.py'
tar -czvf result.tar.gz ./result/"${ZIPDIR}"

