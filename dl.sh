#git fetch --all
#git reset --hard origin/master
python 'dl.py'
ZIPDIR=$(ls -t ./result | head -1)
cp ./param.json ./result/"${ZIPDIR}"/param.json
tar -czvf result.tar.gz ./result/"${ZIPDIR}"

