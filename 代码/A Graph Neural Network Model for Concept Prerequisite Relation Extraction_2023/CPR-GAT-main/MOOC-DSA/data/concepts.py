import pdb
import time



def main():

    with open("./data/CoreConcepts_DSA",'r')as f \
        ,open("./data/concepts.txt",'w') as f1:
        concepts = f.readlines()

        all_concepts = []
        for concept in concepts:
            c = concept[:-1].split("::;")
            all_concepts.append(c)

        concepts = [item for each_list in all_concepts for item in each_list]

        for concept in concepts:
            f1.write("%s\n"%concept)
        



if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	total_time = end_time - start_time
	print("total time: {}".format(total_time))
    #total time: 0.0004 sec