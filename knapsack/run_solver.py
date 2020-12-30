

if __name__ == "__main__":
    import os
    import solver
    rootDir = "data"
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            print(f'Working on {fname}')
            with open(os.path.join(rootDir,fname), "r") as input_data_file:
                input_data = input_data_file.read()
                # print(solve_it_np(input_data))
                print(solver.solve_it(input_data))
