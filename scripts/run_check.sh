#!/bin/bash
cd /root/autodl-tmp
source ~/.bashrc 2>/dev/null || true
# Try conda environments
for py in ~/miniconda3/bin/python ~/anaconda3/bin/python /opt/conda/bin/python; do
    if [ -f "$py" ]; then
        echo "Found: $py"
        $py -c "from alignn.pretrained import get_pretrained_models; m=get_pretrained_models(); ks=list(m.keys()); open('/root/autodl-tmp/models_list.txt','w').write('\n'.join(ks))"
        echo "Done"
        exit 0
    fi
done
echo "No python found"
