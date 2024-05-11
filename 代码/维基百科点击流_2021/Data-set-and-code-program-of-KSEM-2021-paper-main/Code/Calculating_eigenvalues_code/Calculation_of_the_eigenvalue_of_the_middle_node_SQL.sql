----------------------------------------------------------------����ֵɸѡ----------------------------------------------------------
--һ���ڵ�ɸѡ
select	a.A,a.B,isNULL(b.weight,0) as 'weight',isNULL(c.backwardweight,0) as 'backwardweight',(isNULL(b.weight,0)+isNULL(c.backwardweight,0)) as 'SUM',ABS(isNULL(b.weight,0)-isNULL(c.backwardweight,0)) as 'Diff',
isNULL(d.[Sum of all transitionsA],0) as 'Sum of all transitions',(d.[Sum of all transitionsA]/isNULL(d.all_edgesA,1)) as 'Mean of weights',
(isNULL(b.weight,0)/isNULL(d.[Sum of all transitionsA],1)) as 'Normalized weight',
isNULL((c.backwardweight/e.[Sum of all transitionsB]),0) as 'Normalized backward weight',
(isNULL(b.weight,0)-(d.[Sum of all transitionsA]/d.all_edgesA)) as 'Weight greater than mean',
(isNULL(c.backwardweight,0)-isNULL((e.[Sum of all transitionsB]/e.all_edgesB),0)) as 'Backward weight greater than mean'
from
(
select A,B
from [Parallel Postulate(һ���ڵ�)����]
)a
left join
(
--��ɸѡ����Сֵ�е����ֵ
select distinct a.A1,a.B2,MAX(a.�����) as 'weight'
from 
(
--����ɸѡ��A-B,B-C�е���Сֵ
select [Parallel Postulate(һ���ڵ�)����].A as 'A1',a.B as 'B1',b.A as 'A2',[Parallel Postulate(һ���ڵ�)����].B as 'B2',
(
SELECT   
   CASE   
      WHEN a.�����>=b.����� THEN b.�����   
      WHEN a.�����<b.����� THEN a.�����   
   END
) as '�����'

from [Parallel Postulate(һ���ڵ�)����],clickstream a,clickstream b
where REPLACE(a.A,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].A,'_',' ') and  a.B = b.A   
and REPLACE(b.B,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].B,'_',' ')
group by  [Parallel Postulate(һ���ڵ�)����].A,a.B,b.A,[Parallel Postulate(һ���ڵ�)����].B,a.�����,b.�����
)a
group by a.A1,a.B2
)b
on a.A = b.A1 and a.B = b.B2
left join
(
--��ɸѡ����Сֵ�е����ֵ
select distinct a.A1,a.B2,MAX(a.�����) as 'backwardweight'
from 
(
--����ɸѡ��A-B,B-C�е���Сֵ
select [Parallel Postulate(һ���ڵ�)����].A as 'A1',a.B as 'B1',b.A as 'A2',[Parallel Postulate(һ���ڵ�)����].B as 'B2',
(
SELECT   
   CASE   
      WHEN a.�����>=b.����� THEN b.�����   
      WHEN a.�����<b.����� THEN a.�����   
   END
) as '�����'

from [Parallel Postulate(һ���ڵ�)����],clickstream a,clickstream b
where REPLACE(a.A,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].B,'_',' ') and  a.B = b.A   
and REPLACE(b.B,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].A,'_',' ')
group by  [Parallel Postulate(һ���ڵ�)����].A,a.B,b.A,[Parallel Postulate(һ���ڵ�)����].B,a.�����,b.�����
)a
group by a.A1,a.B2
)c
on a.A	 = c.A1 and a.B = c.B2
left join
(
select [Parallel Postulate(һ���ڵ�)����].A,SUM(�����) as 'Sum of all transitionsA',COUNT(*)  as 'all_edgesA'
from [Parallel Postulate(һ���ڵ�)����],clickstream
where REPLACE(clickstream.A,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].A,'_',' ')
group by  [Parallel Postulate(һ���ڵ�)����].A
)d
on a.A = d.A
left join
(
select [Parallel Postulate(һ���ڵ�)����].B,SUM(�����) as 'Sum of all transitionsB',COUNT(*)  as 'all_edgesB'
from [Parallel Postulate(һ���ڵ�)����],clickstream
where REPLACE(clickstream.A,'_',' ') = REPLACE([Parallel Postulate(һ���ڵ�)����].B,'_',' ')
group by  [Parallel Postulate(һ���ڵ�)����].B
)e
on a.B = e.B
where c.backwardweight!=0;
