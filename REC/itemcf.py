#-*- coding: utf-8 -*-
import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

random.seed(0)

'''新用户&时效性优与usercf'''
class ItemBasedCF(object):

    def __init__(self):
        self.trainset = {}

        self.n_sim_product = 20
        self.n_rec_product = 10

        self.product_sim_mat = {}
        self.product_popular = {}
        self.product_count = 0


    @staticmethod
    def loadfile(filename):
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
        fp.close()

    def generate_dataset(self, filename):
        for i,line in enumerate(self.loadfile(filename)):
          if i%100==0:
            user, product, rating, _ = line.split('::')
            self.trainset.setdefault(user, {})
            self.trainset[user][product] = int(rating)


    def calc_product_sim(self):
        ''' 计算产品相似矩阵 '''

        for user, products in self.trainset.items():
            for product in products:
                '''产品流行度矩阵'''
                if product not in self.product_popular:
                    self.product_popular[product] = 0
                self.product_popular[product] += 1

        '''产品总数'''
        self.product_count = len(self.product_popular)
        print('total product number = %d' % self.product_count, file=sys.stderr)

        '''产品倒排矩阵,产品prod1,产品prod1 在一个以上用户那里有过行为开始计数'''

        itemsim_mat = self.product_sim_mat
        for user, products in self.trainset.items():
            for prod1 in products:
                itemsim_mat.setdefault(prod1, defaultdict(int))
                for prod2 in products:
                    if prod1 == prod2:
                        continue
                    itemsim_mat[prod1][prod2] += 1

        '''计算产品相识度'''
        for prod1, related_products in itemsim_mat.items():
            for prod2, count in related_products.items():
                itemsim_mat[prod1][prod2] = count / math.sqrt(
                    self.product_popular[prod1] * self.product_popular[prod2])

    def recommend(self, user):
        ''' 在k个相似产品中推荐N个 '''
        K = self.n_sim_product
        N = self.n_rec_product
        rank = {}
        watched_products = self.trainset[user]

        for product, rating in watched_products.items():
            for related_product, similarity_factor in sorted(self.product_sim_mat[product].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_product in watched_products:
                    continue
                rank.setdefault(related_product, 0)
                '''
                p(u,i)=∑w(i,j)r(u,j)
                为用户U对未接触过的产品i的感兴趣程度，w(i,j)为产品相识度。r(u,j)表示用户对产品j的行为得分。
                求和的基数是 S(i,k)与N(u)的交集，S(i,k)表示和物品i最相似的k个物品，N(u)表示用户u产生过行为的物品集合
                '''
                rank[related_product] += similarity_factor * rating

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_product

        hit = 0
        rec_count = 0
        test_count = 0

        all_rec_products = set()

        popular_sum = 0

        for i, user in enumerate(self.trainset):
          if i % 1000 == 0:

            test_products = self.trainset.get(user, {})
            rec_products = self.recommend(user)
            # print('真实： ' ,test_products)
            # print('推荐： ',rec_products)

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
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(ratingfile)
    itemcf.calc_product_sim()
    itemcf.evaluate()
    for i, user in enumerate(itemcf.trainset):
        if i% 1000==0:
            rec = itemcf.recommend(user)
            print(user,'--> ',rec)
