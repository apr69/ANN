import numpy as np

input_pattern1 = np.array([1,-1,-1,1,-1,-1]).reshape(6,1)
input_pattern2 = np.array([1,1,-1,1,1,-1]).reshape(6,1)
input_pattern3 = np.array([-1,1,-1,1,-1,1]).reshape(6,1)
input_pattern4 = np.array([1,-1,1,1,-1,1]).reshape(6,1)


output_pattern1 = np.array([1,-1,-1]).reshape(3,1)
output_pattern2 = np.array([1,1,-1]).reshape(3,1)
output_pattern3 = np.array([-1,1,-1]).reshape(3,1)
output_pattern4 = np.array([1,-1,1]).reshape(3,1)


weight1 = input_pattern1@output_pattern1.T
weight2 = input_pattern2@output_pattern2.T
weight3 = input_pattern3@output_pattern3.T
weight4 = input_pattern4@output_pattern4.T

print("Weight1 : \n",weight1)
print("\nWeight2 : \n",weight2)
print("\nWeight3 : \n",weight3)
print("\nWeight4 : \n",weight4)

def sign(temp):
  pattern = []
  for i in temp:
    if i > 0:
      pattern.append(1)
    else:
      pattern.append(-1)

  return pattern


temp1 = weight1.T@input_pattern1
print(temp1)
temp2 = weight2.T@input_pattern2
print(temp2)
temp3 = weight3.T@input_pattern3
print(temp3)
temp4 = weight4.T@input_pattern4
print(temp4)


print("Pattern obtained for 1st input : ",sign(temp1))
print("Pattern obtained for 2nd input : ",sign(temp2))
print("Pattern obtained for 3rd input : ",sign(temp3))
print("Pattern obtained for 4th input : ",sign(temp4))


#For Obtaining the sequence of input

temp5 = weight1@output_pattern1
print(temp5)
temp6 = weight2@output_pattern2
print(temp6)
temp7 = weight3@output_pattern3
print(temp7)
temp8 = weight4@output_pattern4
print(temp8)



print("Sequence obtained for 1st pattern : ",sign(temp5))
print("Sequence obtained for 2nd pattern : ",sign(temp6))
print("Sequence obtained for 3rd pattern : ",sign(temp7))
print("Sequence obtained for 4th pattern : ",sign(temp8))
