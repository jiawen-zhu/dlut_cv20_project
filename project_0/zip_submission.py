import os
import shutil
import yaml

def copy_directory(src, dest):
  try:
    shutil.copytree(src, dest)
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
  except OSError as e:
    print('Directory not copied. Error: %s' % e)

shutil.rmtree('temp_submission', ignore_errors=True)
os.mkdir('temp_submission')
dir_list = yaml.load(open('.zip_dir_list.yml'))['directories']
for dir_name in dir_list:
	copy_directory(dir_name, '/'.join(['temp_submission', dir_name]))
shutil.make_archive('submission', 'zip', 'temp_submission')
shutil.rmtree('temp_submission', ignore_errors=True)