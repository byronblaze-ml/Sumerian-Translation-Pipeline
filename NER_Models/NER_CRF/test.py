from NER_CRF import NER_CRF

def savefile(filename,LIST):
    with open(filename, 'w') as f:
        for line in LIST:
            f.write("%s\n" % line)

NER=NER_CRF()
#savefile('ATF_OUTPUT/pipeline2.txt',NER.predict())
NER.predict(output='ATF_OUTPUT/pipeline2.txt')