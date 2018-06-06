all:
	@python setup.py build_ext --inplace
	@rm -rf build
	@cp src/python/__init__.py ./
	@cp src/python/mtcnn.py ./

test:
	@python src/python/test.py

.PHONY: clean
clean:
	@rm -rf build
	@rm -f *.so *.pyc *.png
	@rm -f __init__.py mtcnn.py demo.py
