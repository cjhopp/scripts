#!/usr/bin/python
"""
Here we're taking zipped sc3ml files from Steve, extracting and converting
them to QML via an xslt
"""

def sc3ml2qml(zipdir, outdir, stylesheet):
    import os
    import fnmatch

    raw_files = []
    # raw_dir = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/xsl_test/sc3ml_test'
    for root, dirnames, filenames in os.walk(zipdir):
        for filename in fnmatch.filter(filenames, '*.xml.zip'):
            raw_files.append(os.path.join(root, filename))
    # Running sczip from SC3
    # os.chdir('/home/chet/seiscomp3/lib/')
    for afile in raw_files:
        name = afile.rstrip('.xml.zip')
        cmd_str = ' '.join(['/home/chet/seiscomp3/bin/sczip', '-d', afile,
                            '-o', name])
        os.system(cmd_str)
        # Convert sc3ml to QuakeML
        # Put new files in separate directory
        new_name = ''.join([outdir, os.path.basename(afile).rstrip(),
                            '_QML.xml'])
        cmd_str2 = ' '.join(['xsltproc', '-o', new_name,
                             stylesheet, afile])
        os.system(cmd_str2)
    #Remove all '#' from QuakeML (shady way of circumventing validation issues)
    # qml_files = glob('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/*QML.xml')
    # for one_file in qml_files:
    #     command = "sed -i 's/#//g' " + one_file
    #     os.system(command)
    return
