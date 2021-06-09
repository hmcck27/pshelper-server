from multiprocessing.connection import Client
text = u"칠개의 자연수가 주어질 때, 이들 중 홀수인 자연수들을 모두 골라 그 합을 구하고, 고른 홀수들 중 최솟값을 찾는 프로그램을 작성하시오.예를 들어, 칠개의 자연수 일십이, 칠십칠, 삼십팔, 사십일, 오십삼, 구십이, 팔십오가 주어지면 이들 중 홀수는 칠십칠, 사십일, 오십삼, 팔십오이므로 그 합은칠십칠 + 사십일 + 오십삼 + 팔십오 = 이백오십육이 되고,사십일 < 오십삼 < 칠십칠 < 팔십오이므로 홀수들 중 최솟값은 사십일이 된다."

address = ('localhost', 7000)
conn = Client(address)
conn.send('안녕')
# can also send arbitrary objects:
# conn.send(['a', 2.5, None, int, sum])
conn.close()