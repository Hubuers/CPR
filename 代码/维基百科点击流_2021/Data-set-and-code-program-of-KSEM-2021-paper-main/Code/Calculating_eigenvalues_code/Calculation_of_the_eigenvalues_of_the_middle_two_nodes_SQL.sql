----------------------------------------------------------------����ֵɸѡ----------------------------------------------------------
--�����ڵ�ɸѡ
select	a.A,a.B,isNULL(b.weight,0) as 'weight',isNULL(c.backwardweight,0) as 'backwardweight',(isNULL(b.weight,0)+isNULL(c.backwardweight,0)) as 'SUM',ABS(isNULL(b.weight,0)-isNULL(c.backwardweight,0)) as 'Diff',
isNULL(d.[Sum of all transitionsA],0) as 'Sum of all transitions',(d.[Sum of all transitionsA]/isNULL(d.all_edgesA,1)) as 'Mean of weights',
(isNULL(b.weight,0)/isNULL(d.[Sum of all transitionsA],1)) as 'Normalized weight',
isNULL((c.backwardweight/e.[Sum of all transitionsB]),0) as 'Normalized backward weight',
(isNULL(b.weight,0)-(d.[Sum of all transitionsA]/d.all_edgesA)) as 'Weight greater than mean',
(isNULL(c.backwardweight,0)-isNULL((e.[Sum of all transitionsB]/e.all_edgesB),0)) as 'Backward weight greater than mean'
from
(
select A,B
from [physics����(�����ڵ�)]
)a
left join
(
--��ɸѡ����Сֵ�е����ֵ
select distinct a.A1,a.B2,MAX(a.�����) as 'weight'
from 
(
--����ɸѡ��A1-B,B-C,C--A2�е���Сֵ
--�����ڵ����һ���ڵ�����
--����ɸѡ��A-B,B-C�е���Сֵ

select distinct a.A1,a.B1,c.a,c.B,b.A2,b.B2,(
SELECT   
   CASE   
      WHEN a.�����>=b.����� THEN b.�����   
      WHEN a.�����<b.����� THEN a.�����   
   END
) as '�����'
from 
(
select [physics����(�����ڵ�)].A as 'A1',a.B as 'B1',a.�����
from [physics����(�����ڵ�)],clickstream a
where REPLACE(a.A,'_',' ') = REPLACE([physics����(�����ڵ�)].A,'_',' ') 
group by  [physics����(�����ڵ�)].A,a.B,a.�����
)a 
left join
(
select c.A,c.B
from clickstream c
group by c.A,c.B
)c
on a.B1 = c.A
left join
(
select b.B as 'A2',[physics����(�����ڵ�)].B as 'B2',b.�����
from [physics����(�����ڵ�)],clickstream b
where REPLACE(b.B,'_',' ') = REPLACE([physics����(�����ڵ�)].B,'_',' ') 
group by b.B,[physics����(�����ڵ�)].B,b.�����
)b
on c.B = b.A2
group by a.A1,a.B1,c.a,c.B,b.A2,b.B2,a.�����,b.�����
)a
group by a.A1,a.B2
)b
on a.A = b.A1 and a.B = b.B2

left join
(
--��ɸѡ����Сֵ�е����ֵ
select distinct a.B2,a.A1,MAX(a.�����) as 'backwardweight'
from 
(
--����ɸѡ��A1-B,B-C,C--A2�е���Сֵ
--�����ڵ����һ���ڵ�����
--����ɸѡ��A-B,B-C�е���Сֵ

select distinct  a.A1,a.B1,c.a,c.B,b.A2,b.B2,(
SELECT   
   CASE   
      WHEN a.�����>=b.����� THEN b.�����   
      WHEN a.�����<b.����� THEN a.�����   
   END
) as '�����'
from 
(
select [physics����(�����ڵ�)].B as 'A1',a.B as 'B1',a.�����
from [physics����(�����ڵ�)],clickstream a
where REPLACE(a.A,'_',' ') = REPLACE([physics����(�����ڵ�)].B,'_',' ') 
group by  [physics����(�����ڵ�)].B,a.B,a.�����
)a 
left join
(
select c.A,c.B
from clickstream c
group by c.A,c.B
)c
on a.B1 = c.A
left join
(
select b.B as 'A2',[physics����(�����ڵ�)].A as 'B2',b.�����
from [physics����(�����ڵ�)],clickstream b
where REPLACE(b.B,'_',' ') = REPLACE([physics����(�����ڵ�)].A,'_',' ') 
group by b.B,[physics����(�����ڵ�)].A,b.�����
)b
on c.B = b.A2
group by a.A1,a.B1,c.a,c.B,b.A2,b.B2,a.�����,b.�����
)a
group by a.A1,a.B2
)c
on a.A = c.B2 and a.B = c.A1

left join
(
select [physics����(�����ڵ�)].A,SUM(�����) as 'Sum of all transitionsA',COUNT(*)  as 'all_edgesA'
from [physics����(�����ڵ�)],clickstream
where REPLACE(clickstream.A,'_',' ') = REPLACE([physics����(�����ڵ�)].A,'_',' ')
group by  [physics����(�����ڵ�)].A
)d
on a.A = d.A
left join
(
select [physics����(�����ڵ�)].B,SUM(�����) as 'Sum of all transitionsB',COUNT(*)  as 'all_edgesB'
from [physics����(�����ڵ�)],clickstream
where REPLACE(clickstream.A,'_',' ') = REPLACE([physics����(�����ڵ�)].B,'_',' ')
group by  [physics����(�����ڵ�)].B
)e
on a.B = e.B
where b.weight!=0;
