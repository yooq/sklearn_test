#-*- coding: utf-8 -*-
import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

random.seed(0)

'''有新物品加入时，调用该算法'''
class UserBasedCF(object):
    def __init__(self):
        self.trainset = {}
        self.n_sim_user = 20 #相似用户数量
        self.n_rec_product = 10 #推荐产品数量
        self.user_sim_mat = {}
        self.product_popular = {}
        self.product_count = 0
        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended product number = %d' %
               self.n_rec_product, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' 读取数据'''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename):
        for i,line in enumerate(self.loadfile(filename)):
          if i%100==0:
            user, product, rating, _ = line.split('::')
            self.trainset.setdefault(user, {})
            self.trainset[user][product] = int(rating)


    def calc_user_sim(self):
        '''计算相似用户'''
        # build inverse table for item-users
        # key=productID, value=list of userIDs who have seen this product
        print ('building product-users inverse table...', file=sys.stderr)
        '''定义一个倒排字典 p-u'''
        product2users = dict()
        for user, products in self.trainset.items():
            for product in products:
                '''以产品为key,用户为value，其结构为{key:[value1,value2,value3...]}'''
                if product not in product2users:
                    product2users[product] = set()
                product2users[product].add(user)
                '''计算产品受欢迎程度---用户群对改产品的行为次数'''
                if product not in self.product_popular:
                    self.product_popular[product] = 0
                self.product_popular[product] += 1
        print ('build product-users inverse table succ', file=sys.stderr)

        '''产品数量'''
        self.product_count = len(product2users)
        print ('total product number = %d' % self.product_count, file=sys.stderr)

        '''用户倒排矩阵'''
        usersim_mat = self.user_sim_mat
        print ('building user co-rated products matrix...', file=sys.stderr)

        for product, users in product2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated products matrix succ', file=sys.stderr)
        '''计算用户相似矩阵'''
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))

                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',file=sys.stderr)
        print ('Total similarity factor number = %d' %simfactor_count, file=sys.stderr)


    def recommend(self, user):
        ''' 寻找k个相似用户，推荐n个产品 '''
        K = self.n_sim_user
        N = self.n_rec_product
        rank = dict()
        watched_products = self.trainset[user]

        # print('user: ',user,'watched ',watched_products)
        # print(self.user_sim_mat[user].items())

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),key=itemgetter(1), reverse=True)[0:K]:

            for product in self.trainset[similar_user]:

                '''如果产品已被用户购买过，则不参加推荐，得分为0'''
                if product in watched_products:
                    continue
                '''否则计算相似用户中，该产品的累积得分，相似用户中大多购买过该产品，则有理由相信当前用户对该物品兴趣大'''
                rank.setdefault(product, 0)
                rank[product] += similarity_factor
        # print('根据用户相似推荐10个产品：')
        # print(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        '''评估，准确率，召回率，覆盖率，流行度'''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_product

        hit = 0
        rec_count = 0
        test_count = 0

        all_rec_products = set()

        popular_sum = 0

        for i, user in enumerate(self.trainset):
           if i%1000==0:
            test_products = self.trainset.get(user, {})
            rec_products = self.recommend(user)
            # print('真实：',test_products)
            # print('推荐：',rec_products)
            for product, _ in rec_products:
                if product in test_products:
                    hit += 1
                all_rec_products.add(product)
                popular_sum += math.log(1 + self.product_popular[product])
            rec_count += N
            test_count += len(test_products)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_products) / (1.0 * self.product_count)
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    usercf = UserBasedCF()
    usercf.generate_dataset(ratingfile)
    usercf.calc_user_sim()
    usercf.evaluate()

    '''预测或推荐'''

    for i, user in enumerate(usercf.trainset):
        if i% 1000==0:
            rec = usercf.recommend(user)
            print(user,'--> ',rec)

