import shutil


def update_index_md():
    shutil.copyfile('../README.md', 'sources/index.md')

if __name__ == "__main__":
    update_index_md()
